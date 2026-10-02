import os
import argparse
import yaml
import skimage
from skimage import transform
import numpy as np
from tqdm import tqdm
import sys

# Per importare i moduli da 'src' anche se siamo in 'scripts'
# Aggiunge la cartella superiore al path di Python
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation.sam_loader import load_sam_model
from src.augmentation import sam_methods

def run_processing(config):
    print("Configuration loaded:", config.keys())
    dataset_root = config['paths']['dataset_root']
    output_root = config['paths']['output_root']

    for sam_version in config['sam']['versions']:
        checkpoints_root = config['paths']['checkpoints_root']

        if sam_version == 1:
            path_to_sam_checkpoint=os.path.join(checkpoints_root,config['sam']['checkpoints']['v1'])
            sam_folder = "sam1"
        elif sam_version == 2:
            path_to_sam_checkpoint=os.path.join(checkpoints_root,config['sam']['checkpoints']['v2'])
            # folder name built from the stability threshold, e.g. 0.85 -> "sam2_stab085"
            stability_score = config['sam']['thresholds']['v2']['stability_score']
            sam_folder = "sam2_stab" + f"{stability_score:.2f}".replace(".", "")

        mask_generator = load_sam_model(sam_version, path_to_sam_checkpoint, config['sam'])
        
        print(f"------ Running generationwith SAM v{sam_version} ------")

        for cur_dir in config['datasets']['folders']:
                
            folder_path = os.path.join(dataset_root, cur_dir, "images")
            saving_path = os.path.join(output_root, sam_folder, cur_dir)
            
            print(f"Processing dataset folder: {folder_path}")
            print(f"Saving SAM predictions and priors to: {saving_path}")

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

                std_dim = tuple(config['image_processing']['resize_dim'])

                original_size=tI.shape
                resized_flag=False

                #resize the image if too large
                if original_size[0]>=config['image_processing']['resize_threshold'] or  original_size[1]>=config['image_processing']['resize_threshold']:
                    resized_flag=True
                    tI = transform.resize(tI, std_dim, anti_aliasing=True)
                    tI=tI.astype(np.float32)
                    print(filename, ": resized to: ", tI.shape, "FIXED")

                #replace here with the wanted augmentation method
                segPrior, boundaryPrior = sam_methods.raw_SAM_generation(tI, mask_generator)

                if resized_flag:
                    segPrior = transform.resize(segPrior, original_size, anti_aliasing=True)
                    boundaryPrior = transform.resize(boundaryPrior, original_size, anti_aliasing=True)
                
                segPrior_path = os.path.join(saving_path, "segPrior")
                os.makedirs(segPrior_path, exist_ok=True)
                boundaryPrior_path = os.path.join(saving_path, "boundaryPrior")
                os.makedirs(boundaryPrior_path, exist_ok=True)

                # Save the generated maps to the specified folder
                np.save(os.path.join(segPrior_path, filename + ".npy"), segPrior)
                np.save(os.path.join(boundaryPrior_path, filename + ".npy"), boundaryPrior)

                count = count + 1
                print("(-) ",cur_dir, ": ", count,"/",len(file_names))


if __name__ == "__main__":

    # Find directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # build the config file path
    config_path = os.path.abspath(os.path.join(current_dir, '../configs/sam_generation.yaml'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path, help='Path to the config file.')
    opt = parser.parse_args()

    if not os.path.exists(opt.config):
        raise FileNotFoundError(f"Config file not found: {opt.config}")
    
    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    run_processing(config)
