import torch
import os
import argparse
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset_with_aux # ,get_loader_with_aux
import numpy as np
# import random
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import yaml

# ------- YAML config ---------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Config(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)
# ----------------------------------

def test(model, opt, dataset):

    data_path = os.path.join(opt.paths.datasets_root, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)

    aux_folder= os.path.join(opt.paths.aux_root_base, f"sam{opt.experiment.sam_version}" , opt.experiment.method, dataset)
    aux_root = os.path.join(aux_folder, "images/")

    logits_dir = os.path.join(opt.paths.logits_dir, opt.model_name, dataset)
    os.makedirs(logits_dir, exist_ok=True)

    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset_with_aux(image_root, gt_root, aux_root, opt.testing.testsize)
    DSC = 0.0
    for i in range(num1):
        image, gt, aux,name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        aux=aux.cuda()

        res, res1  = model(image)
        res = F.interpolate(res + res1 , size=gt.shape, mode='bilinear', align_corners=False)
        res_logits = res.data.cpu().numpy().squeeze()
        res = res.sigmoid().data.cpu().numpy().squeeze()
        #res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        augres, augres1  = model(aux)
        augres = F.interpolate(augres + augres1 , size=gt.shape, mode='bilinear', align_corners=False)
        augres_logits = augres.data.cpu().numpy().squeeze()
        augres = augres.sigmoid().data.cpu().numpy().squeeze()
        #res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        indicator1=np.mean(np.abs(res-0.5))
        indicator2=np.mean(np.abs(augres-0.5))
        if indicator1>indicator2:
            input = res
            logits = res_logits
        else:
            input = augres
            logits = augres_logits

        np.save(os.path.join(logits_dir, name.replace('.png', '.npy')), logits)

        to_pil = transforms.ToPILImage()
        pil_img = to_pil(input)

        save_root = os.path.join(opt.paths.prediction_dir, opt.model_name)
        save_dir = os.path.join(save_root, dataset)
        
        os.makedirs(save_dir, exist_ok=True)

        pil_img.save(os.path.join(save_dir, name))
        
        target = np.array(gt)
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config',type=str, default='../../../configs/polypPVT_aux.yaml')
    parser.add_argument('--test_dataset', type=str , default=None)
    parser.add_argument('--model_pth', type=str , default=None)
    parser.add_argument('--sam_version' , type=int , default=None, help='Version of SAM to use for augmentation data (e.g., 1 or 2). If not specified, it will use the default version defined in the configuration file.')
    parser.add_argument('--method', type=str, default=None, help='Method used for generating augmentation data. If not specified, it will use the default method defined in the configuration file.')
    
    args = parser.parse_args()

    cfg_data = load_config(args.config)
    opt = Config(cfg_data)

    # CLI overrides
    if args.model_pth is not None:
        model_pth = os.path.join(opt.paths.models_dir, args.model_pth)
    else:           
        model_pth = os.path.join(opt.paths.models_dir, opt.testing.test_checkpoint)

    if args.sam_version is not None:
        opt.experiment.sam_version = args.sam_version
    if args.method is not None:
        opt.experiment.method = args.method

    test_datasets = [args.test_dataset] if args.test_dataset else opt.datasets.test

    model_name = os.path.splitext(os.path.basename(model_pth))[0]
    opt.model_name = model_name

    model = PolypPVT()
    model.load_state_dict(torch.load(model_pth))
    model.cuda()

    scores = []
    for dataset in test_datasets:
        score = test(model, opt, dataset)
        scores.append(score)
        print(dataset, score)
    print("center_m", sum(scores)/len(scores))

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()