import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
# from scipy import misc
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
import cv2
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

def main():
     #1 parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../../../configs/polypvt_vanilla.yaml', help='path to config file')
    
    #2 ovveride for launcher
    parser.add_argument('--model_pth', type=str, default=None, help='Override model pth')
    parser.add_argument('--test_dataset', type=str, default=None, help='Override test dataset path')
    parser.add_argument('--save_path', type=str, default=None, help='Override save path for predictions')

    #3 parse args
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERRORE: FILE CONFIGURAZIONE NON TROVATO: {args.config}")
        exit(1)

    cfg_data = load_config(args.config)
    opt = Config(cfg_data)

    # Override with command line arguments if provided
    model_pth = args.model_pth if args.model_pth is not None else os.path.join(opt.paths.models_dir, opt.testing.test_checkpoint)
    test_datasets = [args.test_dataset] if args.test_dataset is not None else opt.datasets.test
    save_path = args.save_path if args.save_path is not None else opt.paths.prediction_dir
    testsize = opt.testing.testsize

    #4 load model
    model = PolypPVT()
    model.load_state_dict(torch.load(model_pth))
    model.cuda()
    model.eval()   

    for _data_name in test_datasets:

        data_path = os.path.join(opt.paths.datasets_root, _data_name)
        save_dir = os.path.join(save_path, _data_name)

        os.makedirs(save_dir, exist_ok=True)
        image_root = os.path.join(data_path, 'images/')
        gt_root = os.path.join(data_path, 'masks/')
        num1 = len(os.listdir(gt_root))

        test_loader = test_dataset(image_root, gt_root, testsize)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            P1,P2 = model(image)
            res = F.interpolate(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(os.path.join(save_dir, name), res*255)
        print(_data_name, 'Finish!')

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()      