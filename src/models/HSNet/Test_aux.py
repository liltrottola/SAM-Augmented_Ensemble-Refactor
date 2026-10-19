import torch
import os
import argparse
from lib.pvt_aux import HSNet_with_aux
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

'''
nothing random happening:

The model weights are fixed (loaded from a .pth file)
The test data is loaded in order, no shuffling
The inference is purely deterministic math

seednumber=10
random.seed(seednumber)     # python random generator
np.random.seed(seednumber)  # numpy random generator

torch.manual_seed(seednumber)
torch.cuda.manual_seed_all(seednumber)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''

def test(model, opt, dataset):
    data_path = os.path.join(opt.paths.datasets_root, dataset)
    image_root = os.path.join(data_path, "images/")
    gt_root = os.path.join(data_path, "masks/")

    aux_folder= os.path.join(opt.paths.aux_root_base, opt.experiment.aux_source , opt.experiment.method, dataset)
    aux_root = os.path.join(aux_folder, "images/")

    logits_dir = os.path.join(opt.paths.logits_dir, opt.model_name, dataset)
    os.makedirs(logits_dir, exist_ok=True)

    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset_with_aux(image_root, gt_root, aux_root, opt.testing.testsize)

    # The loader pairs image and aux by position in the sorted lists (and here it does not
    # even check the counts), so check that the names match one to one
    print(f"{dataset} - aux images: {aux_root} ({len(test_loader.aux)} files)")
    image_names = [os.path.splitext(os.path.basename(f))[0] for f in test_loader.images]
    aux_names = [os.path.splitext(os.path.basename(f))[0] for f in test_loader.aux]
    if image_names != aux_names:
        mismatch = [(i, a) for i, a in zip(image_names, aux_names) if i != a]
        print(f"ERROR: IMAGE/AUX NAMES DO NOT MATCH in {dataset} ({len(image_names)} images, {len(aux_names)} aux, "
              f"{len(mismatch)} mismatched pairs), first: {mismatch[:5]}")
        exit(1)

    DSC = 0.0
    counter_aux = 0
    counter_img = 0
    for i in range(num1):
        image, gt, aux,name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        aux=aux.cuda()

        res,res1,res2,res3,_,_,_,_ = model(image)
        res = F.interpolate(res + res1 + res2 + res3, size=gt.shape, mode='bilinear', align_corners=False)
        
        res_logits = res.data.cpu().numpy().squeeze()
        
        res = res.sigmoid().data.cpu().numpy().squeeze()
        #res = (res - res.min()) / (res.max() - res.min() + 1e-8)    
        #tmp = (res - res.min()) / (res.max() - res.min() + 1e-8)    

        indicator1=np.mean(np.abs(res-0.5))
        #TODO:anche qui image + aux
        Ares,Ares1,Ares2,Ares3,_,_,_,_ = model(aux)
        Ares = F.interpolate(Ares + Ares1 + Ares2 + Ares3, size=gt.shape, mode='bilinear', align_corners=False)
        
        ares_logits = Ares.data.cpu().numpy().squeeze()

        Ares = Ares.sigmoid().data.cpu().numpy().squeeze()
        #Ares = (Ares - Ares.min()) / (Ares.max() - Ares.min() + 1e-8)
        #tmp = (Ares - Ares.min()) / (Ares.max() - Ares.min() + 1e-8)

        indicator2=np.mean(np.abs(Ares-0.5))            
        if indicator1>indicator2:
            input = res
            logits = res_logits
            counter_img=counter_img+1
        else:
            input = Ares
            logits = ares_logits
            counter_aux=counter_aux+1

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
    #print("counter_aux/counter_img: ", counter_aux/counter_img)
    return DSC / num1

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config',type=str, default='../../../configs/hsnet_aux.yaml')
    parser.add_argument('--test_dataset', type=str , default=None)
    parser.add_argument('--model_pth', type=str , default=None)
    parser.add_argument('--aux_source', type=str, default=None, help='Folder under aux_root_base with the aux images (e.g. sam1, sam2, da3mono-large). If not specified, it will use experiment.aux_source defined in the configuration file.')
    parser.add_argument('--method', type=str, default=None, help='Method used for generating augmentation data. If not specified, it will use the default method defined in the configuration file.')

    args = parser.parse_args()

    cfg_data = load_config(args.config)
    opt = Config(cfg_data)

    # CLI overrides
    if args.model_pth is not None:
        model_pth = os.path.join(opt.paths.models_dir, args.model_pth)
    else:           
        model_pth = os.path.join(opt.paths.models_dir, opt.testing.test_checkpoint)

    if args.aux_source is not None:
        opt.experiment.aux_source = args.aux_source
    if args.method is not None:
        opt.experiment.method = args.method

    test_datasets = [args.test_dataset] if args.test_dataset else opt.datasets.test

    model_name = os.path.splitext(os.path.basename(model_pth))[0]
    opt.model_name = model_name

    model = HSNet_with_aux().cuda()
    model.load_state_dict(torch.load(model_pth))

    scores = []
    for dataset in test_datasets:
        score = test(model, opt, dataset)
        scores.append(score)
        print(dataset, score)
    print("center_m", sum(scores)/len(scores))

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

