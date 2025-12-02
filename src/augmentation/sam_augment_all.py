from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from skimage import transform
import numpy as np
import skimage
import os
import argparse

from lib import *

import yaml


#get command line arguments in an elegant way
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="../../configs/augmentation.yaml", help='Path to the config file.')
opt = parser.parse_args()

try:
    opt = parser.parse_args()
except:
    print("No command line arguments given, using default values")  

try:
    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Config file {opt.config} not found. Using default configuration.")
    config = {} # Empty config if file not found, TODO : set default values here if needed

# Subito dopo with open(...) as f: cfg = yaml.safe_load(f)
print("Configurazione caricata:", config.keys())


AVAILABLE_METHODS = {
    "SAMAug": SAMAug,
    "ourSAMAug": ourSAMAug,
    "RG_segPrior": RG_segPrior,
    "SV_segPrior": SV_segPrior,
    "RG_logits": RG_logits,
    "PCA_segPrior": PCA_segPrior,
}

#User parameters from config file
for sam_version in config['sam']['versions']:
    # Loop methods names from config
    for chosen_aug in config['augmentation']['methods']:

        aug_function = AVAILABLE_METHODS[chosen_aug]

        # Costruisci path salvataggio
        # Nota: dataset_path ora è cfg['paths']['dataset_root']
        dataset_name=os.path.basename(os.path.normpath(config['paths']['dataset_root']))
        saving_path= os.path.join(config['paths']['output_root'], dataset_name, "sam"+str(sam_version) , aug_function.__name__)


        print("sam_version: ", sam_version)
        if sam_version == 1:
            path_to_sam_checkpoint=os.path.join(config['paths']['checkpoints_root'],config['sam']['checkpoints']['v1'])
        elif sam_version == 2:
            path_to_sam_checkpoint=os.path.join(config['paths']['checkpoints_root'],config['sam']['checkpoints']['v2'])

        print("path checkpoints: ", path_to_sam_checkpoint)

        #model choice:
        if sam_version == 2:
            model_cfg = "sam2_hiera_l.yaml"
            sam2 = build_sam2(model_cfg, path_to_sam_checkpoint, device ='cuda', apply_postprocessing=False)
            mask_generator = SAM2AutomaticMaskGenerator(sam2)

            #to load a finetuned model
            #ft_model=torch.load("/.../model.torch")
            #mask_generator.model.load_state_dict(ft_model))
        elif sam_version == 1:
            sam = sam_model_registry["vit_h"](checkpoint=path_to_sam_checkpoint)
            sam.to(device='cuda')
            mask_generator = SamAutomaticMaskGenerator(sam,crop_nms_thresh=0.5,box_nms_thresh=0.5,pred_iou_thresh=0.5)
        else :
            print("(x) Error: insert a valid SAM version (1,2)")

        print("(!) chosen sam version: " , sam_version, "chosen method: ", aug_function.__name__)

        #running wanted augmentation method for each dataset
        for cur_dir in config['datasets']['folders']:
            #path to the folder containing the images to augment
            folder_path = os.path.join(config['paths']['dataset_root'],cur_dir,"images")
            file_names = os.listdir(folder_path)

            count = 0
            for filename in file_names:
                tI = skimage.io.imread(os.path.join(folder_path, filename))

                shape=tI.shape
                # Check if the input image is grayscale (i.e., has only two dimensions: height and width)
                if len(shape) == 2:
                    # Convert the grayscale image to an RGB image by duplicating the single channel across three channels
                    tmp = np.stack([tI] * 3, axis=-1)
                    # Assign the converted image back to tI
                    tI = tmp
                    shape=tI.shape

                #in case of RGB+alpha channel, we remove the alpha channel since is unused
                if shape[2]==4:
                    print(filename,": wrong shape FIXED")
                    tmp=np.empty((shape[0], shape[1], 3), dtype=float)
                    tmp[:,:,0]=tI[:,:,0]
                    tmp[:,:,1]=tI[:,:,1]
                    tmp[:,:,2]=tI[:,:,2]
                    tI=tmp
                
                if tI.dtype == np.float64:
                    tI = (tI * 255).astype(np.uint8)
                    print(filename,": wrong type FIXED")

                std_dim = (1920,1280)
                original_size=tI.shape
                resized_flag=False
                #resize the image if too large
                if original_size[0]>=2000 or original_size[1]>=2000:
                    resized_flag=True
                    tI = transform.resize(tI, std_dim, anti_aliasing=True)
                    tI=tI.astype(np.float32)
                    print(filename, ": resized to: ", tI.shape, "FIXED")

                #replace here with the wanted augmentation method
                aug_img = aug_function(tI, mask_generator)
                
                if resized_flag:
                    aug_img = transform.resize(aug_img, original_size, anti_aliasing=True)

                # Convert to uint8
                aug_img_scaled = skimage.exposure.rescale_intensity(aug_img, in_range='image', out_range=(0, 1))
                aug_img_uint8 = skimage.img_as_ubyte(aug_img_scaled)

                # save result
                full_saving_path= os.path.join(saving_path, cur_dir, "images")
                os.makedirs(full_saving_path, exist_ok=True)
                skimage.io.imsave(os.path.join(full_saving_path, filename), aug_img_uint8)


                count = count + 1
                print("(-) ",cur_dir, ": ", count,"/",len(file_names))
