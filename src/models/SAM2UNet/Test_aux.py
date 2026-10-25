import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
#from scipy import misc
import yaml
from SAM2UNet import SAM2UNet
from utils.dataloader import test_dataset_with_aux   # aux: test loader image + gt + aux instead of test_dataset

from PIL import Image
import torchvision.transforms as transforms

# -------- configuration from yaml file -----------
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Config(object):
    def __init__(self, d):
        for k , v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)
# ------------------------------------------------------------

def main():
    #1 parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../../../configs/sam2unet_aux.yaml', help='path to config file')

    #2 override for launcher
    parser.add_argument('--model_pth', type=str, default=None, help='Override model pth')
    parser.add_argument('--test_dataset', type=str, default=None, help='Override test dataset path')
    parser.add_argument('--save_path', type=str, default=None, help='Override save path for predictions')
    # aux: aux source and method
    parser.add_argument('--aux_source', type=str, default=None, help='Folder under aux_root_base with the aux images (e.g. sam1, sam2, da3mono-large). If not specified, it will use experiment.aux_source defined in the configuration file.')
    parser.add_argument('--method', type=str, default=None, help='Method used for generating augmentation data. If not specified, it will use the default method defined in the configuration file.')

    #3 parse args
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: CONFIGURATION FILE NOT FOUND: {args.config}")
        exit(1)

    cfg_data = load_config(args.config)
    opt = Config(cfg_data)

    # Override with command line arguments if provided
    if args.model_pth is not None:
        model_pth = os.path.join(opt.paths.models_dir, args.model_pth)
    else:           
        model_pth = os.path.join(opt.paths.models_dir, opt.testing.test_checkpoint)
    
    if args.aux_source is not None:
        opt.experiment.aux_source = args.aux_source
    if args.method is not None:
        opt.experiment.method = args.method

    test_datasets = [args.test_dataset] if args.test_dataset is not None else opt.datasets.test
    save_path = args.save_path if args.save_path is not None else opt.paths.prediction_dir
    testsize = opt.testing.testsize

    model_name = os.path.splitext(os.path.basename(model_pth))[0]
    opt.model_name = model_name

    #4 load model
    model = SAM2UNet()   # no hiera checkpoint at test: all weights (frozen encoder included) come from the .pth, as original test.py
    model.load_state_dict(torch.load(model_pth), strict=True)
    model.cuda()
    model.eval()    

    scores = []
    #5 test loop
    for _data_name in test_datasets:
        data_path = os.path.join(opt.paths.datasets_root, _data_name)
        save_dir = os.path.join(save_path, model_name, _data_name)
        os.makedirs(save_dir, exist_ok=True)

        logits_dir = os.path.join(opt.paths.logits_dir, model_name, _data_name)
        os.makedirs(logits_dir, exist_ok=True)

        image_root = os.path.join(data_path, "images/")
        gt_root = os.path.join(data_path, "masks/")
        # aux: aux images of the test set (always the original images, no DA at test)
        aux_root = os.path.join(opt.paths.aux_root_base, opt.experiment.aux_source, opt.experiment.method, _data_name, "images/")
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset_with_aux(image_root, gt_root, aux_root, testsize)

        # aux: the loader pairs image and aux by position in the sorted lists (and here it does not
        # even check the counts), so check that the names match one to one 
        print(f"{_data_name} - aux images: {aux_root} ({len(test_loader.aux)} files)")
        image_names = [os.path.splitext(os.path.basename(f))[0] for f in test_loader.images]
        aux_names = [os.path.splitext(os.path.basename(f))[0] for f in test_loader.aux]
        if image_names != aux_names:
            mismatch = [(i, a) for i, a in zip(image_names, aux_names) if i != a]
            print(f"ERROR: IMAGE/AUX NAMES DO NOT MATCH in {_data_name} ({len(image_names)} images, {len(aux_names)} aux, "
                  f"{len(mismatch)} mismatched pairs), first: {mismatch[:5]}")
            exit(1)

        DSC = 0.0 #to accumulate dice scores

        for i in range(num1):
            image, gt, aux, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            aux = aux.cuda()
            with torch.no_grad():   # as original SAM2-UNet test.py
                res, _, _ = model(image)  # only the first output (final head) is used at test
                augres, _, _ = model(aux)  # aux: same prediction on the aux image
            #original code: res = F.upsample(..) now deprecated, replaced with interpolate
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res_logits = res.data.cpu().numpy().squeeze()
            res = res.sigmoid().data.cpu().numpy().squeeze()

            augres = F.interpolate(augres, size=gt.shape, mode='bilinear', align_corners=False)
            augres_logits = augres.data.cpu().numpy().squeeze()
            augres = augres.sigmoid().data.cpu().numpy().squeeze()

            # aux: keep the more confident prediction (mean distance from 0.5), as PolypPVT Test_aux
            indicator1 = np.mean(np.abs(res - 0.5))
            indicator2 = np.mean(np.abs(augres - 0.5))
            if indicator1 > indicator2:
                logits_np = res_logits
            else:
                res = augres
                logits_np = augres_logits

            #we save the logits of the chosen prediction
            np.save(os.path.join(logits_dir, name.replace('.png', '.npy')), logits_np)

            to_pil = transforms.ToPILImage()
            pil_img = to_pil(res)
            pil_img.save(os.path.join(save_dir, name))

            target = np.array(gt)
            N = gt.shape
            smooth = 1
            input_flat = np.reshape(res, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (res.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC = DSC + dice

        score = DSC / num1
        scores.append(score)

        print(_data_name, 'Finish!')
        print(_data_name, score)
    
    print('Average DSC across datasets:', sum(scores)/len(scores))

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()