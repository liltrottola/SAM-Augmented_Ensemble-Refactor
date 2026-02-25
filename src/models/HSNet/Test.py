import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import yaml
from lib.pvt import HSNet
from utils.dataloader import test_dataset

from PIL import Image
import torchvision.transforms as transforms

# -------- blocco per configurazioni da file yaml -----------
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

    # Override con argomenti da linea di comando
    if args.model_pth is not None:
        opt.model_pth = args.model_pth
    if args.test_dataset is not None:
        opt.test_dataset = args.test_dataset
    if args.save_path is not None:
        opt.save_path = args.save_path  

    
    



if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    '''
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--single', type=str)
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--model_pth', type=str, default='./model_pth/HSNet.pth')
    parser.add_argument('--save_path', type=str, default="./generated")
    parser.add_argument('--test_dataset', type=str, default="../polyp_dataset/TestDataset")

    parser.add_argument('--config', type=str, default='../../../configs/hsnet_vanilla.yaml', help='path to config file')
    
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERRORE: FILE CONFIGURAZIONE NON TROVATO: {args.config}")
        exit(1)

    config = load_config(args.config)
    opt = Config(config)

    model = HSNet()
    model.load_state_dict(torch.load(opt.model_pth))
    model.cuda()
    model.eval()

    dataset_list=['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

    if opt.single=="true":
        dataset_list=[""]


    for _data_name in dataset_list:

        ##### put data_path here #####
        data_path = opt.test_dataset + "/"+ _data_name


        if not os.path.exists(opt.save_path+"/"+_data_name):
            os.makedirs(opt.save_path+"/"+_data_name)
            
        image_root = os.path.join(data_path, "images/")
        gt_root = os.path.join(data_path, "masks/")

        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            P1,P2,P3,P4 = model(image)
            res = F.upsample(P1+P2+P3+P4, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            #res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            to_pil = transforms.ToPILImage()
            pil_img = to_pil(res)
            pil_img.save(opt.save_path+"/"+_data_name+"/"+name)
        print(_data_name, 'Finish!')
'''