import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
#from scipy import misc
import yaml
from lib.pvt_vanilla import HSNet
from utils.dataloader import test_dataset

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
    parser.add_argument('--config', type=str, default='../../../configs/hsnet_vanilla.yaml', help='path to config file')
    
    #2 override for launcher
    parser.add_argument('--model_pth', type=str, default=None, help='Override model pth')
    parser.add_argument('--test_dataset', type=str, default=None, help='Override test dataset path')
    parser.add_argument('--save_path', type=str, default=None, help='Override save path for predictions')

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
    
    test_datasets = [args.test_dataset] if args.test_dataset is not None else opt.datasets.test
    save_path = args.save_path if args.save_path is not None else opt.paths.prediction_dir
    testsize = opt.testing.testsize

    model_name = os.path.splitext(os.path.basename(model_pth))[0]
    opt.model_name = model_name

    #4 load model
    model = HSNet()
    model.load_state_dict(torch.load(model_pth))
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
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, testsize)

        DSC = 0.0 #to accumulate dice scores

        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            P1,P2,P3,P4 = model(image)
            #original code: res = F.upsample(..) now deprecated, replaced with interpolate
            res = F.interpolate(P1+P2+P3+P4, size=gt.shape, mode='bilinear', align_corners=False)

            #we save logits and then apply sigmoid before saving
            logits_np = res.data.cpu().numpy().squeeze()
            np.save(os.path.join(logits_dir, name.replace('.png', '.npy')), logits_np)

            #apply sigmoid to get probabilities before saving as image
            res = res.sigmoid().data.cpu().numpy().squeeze()

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