import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import yaml
from SAM2RFANet import SAM2RFANet
from utils.dataloader import test_dataset
from PIL import Image
import torchvision.transforms as transforms

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../../../configs/sam2rfanet_vanilla.yaml')
    parser.add_argument('--model_pth', type=str, default=None)
    parser.add_argument('--test_dataset', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.config): exit(1)

    cfg_data = load_config(args.config)
    opt = Config(cfg_data)

    if args.model_pth is not None:
        model_pth = os.path.join(opt.paths.models_dir, args.model_pth)
    else:           
        model_pth = os.path.join(opt.paths.models_dir, opt.testing.test_checkpoint)
    
    test_datasets = [args.test_dataset] if args.test_dataset is not None else opt.datasets.test
    save_path = args.save_path if args.save_path is not None else opt.paths.prediction_dir
    testsize = opt.testing.testsize

    model_name = os.path.splitext(os.path.basename(model_pth))[0]
    opt.model_name = model_name

    model = SAM2RFANet()  
    model.load_state_dict(torch.load(model_pth), strict=True)
    model.cuda()
    model.eval()    

    scores = []
    
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

        DSC = 0.0 

        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            with torch.no_grad():   
                # RFANet restituisce 4 logits. Al test si tiene solo il primo (la head finale).
                res, _, _, _ = model(image)

            #we save logits at the network output resolution
            logits_np = res.data.cpu().numpy().squeeze().astype(np.float16)
            np.save(os.path.join(logits_dir, name.replace('.png', '.npy')), logits_np)

            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)

            res = res.sigmoid().data.cpu().numpy().squeeze()
            to_pil = transforms.ToPILImage()
            pil_img = to_pil(res)
            pil_img.save(os.path.join(save_dir, name))

            target = np.array(gt)
            smooth = 1
            input_flat = np.reshape(res, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (res.sum() + target.sum() + smooth)
            dice = float('{:.4f}'.format(dice))
            DSC = DSC + dice

        score = DSC / num1
        scores.append(score)

        print(_data_name, 'Finish! mDice:', score)
    
    print('Average DSC across datasets:', sum(scores)/len(scores))

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

