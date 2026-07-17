import os
import sys
import yaml
import skimage
from skimage import io
import numpy as np

import argparse

# Add the parent directory to the sys.path to allow imports from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation import da1_methods

def run_processing(config):
    print("Configuration loaded:", config.keys())
    dataset_root = config['paths']['dataset_root']
    output_root = config['paths']['output_root']
    source_name = config['paths']['source_name']

    method_list = config['augmentation'].get('methods') or []
    keep_original = config['augmentation'].get('keep_original', True)

    filter_cfg = config['augmentation'].get('foreground_filter', {})
    filter_enabled = filter_cfg.get('enabled', True)
    min_pixels = filter_cfg.get('min_pixels', 100)

    for cur_dir in config['datasets']['folders']:
        #Path input
        image_folder = os.path.join(dataset_root, cur_dir, "images")
        mask_folder = os.path.join(dataset_root, cur_dir, "masks")

        #Path output
        saving_path_images = os.path.join(output_root,"da1", source_name, cur_dir, "images")
        saving_path_masks = os.path.join(output_root, "da1", source_name, cur_dir, "masks")

        print(f"Processing dataset folder: {image_folder}")
        print(f"Saving augmented images to: {saving_path_images}")
        print(f"Saving augmented masks to: {saving_path_masks}")   
        os.makedirs(saving_path_images, exist_ok=True)
        os.makedirs(saving_path_masks, exist_ok=True)

        file_names = os.listdir(image_folder)
        count = 0

        for filename in file_names:
            image_path = os.path.join(image_folder, filename)
            mask_path = os.path.join(mask_folder, filename)

            tI = io.imread(image_path)
            tM = io.imread(mask_path)

            if tM.ndim == 3:
                tM = tM.max(axis=-1)    # 3D RGB is triple → 2D (max per pixel) DA2 methods assumes 2D masks

            # Save the original if requested
            if keep_original:
                io.imsave(os.path.join(saving_path_images, filename), tI)
                io.imsave(os.path.join(saving_path_masks, filename), tM)
                count += 1

            for method_name in method_list:
                if hasattr(da1_methods, method_name):
                    aug_function = getattr(da1_methods, method_name)
                    augmented_image, augmented_mask = aug_function(tI, tM)

                    # Apply the foreground pixel filter if enabled
                    if filter_enabled:
                        if not da1_methods.has_enough_foreground(augmented_mask, min_pixels):
                            print(f"Skipping {filename} with {method_name} due to insufficient foreground pixels.")
                            continue

                    # Save the augmented images and masks
                    aug_filename = f"{os.path.splitext(filename)[0]}_{method_name}{os.path.splitext(filename)[1]}"
                    io.imsave(os.path.join(saving_path_images, aug_filename), augmented_image)
                    io.imsave(os.path.join(saving_path_masks, aug_filename), augmented_mask)

                    count += 1

        print(f"Total augmented images saved for {cur_dir}: {count}")

if __name__ == "__main__":

    # Find directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to config file
    config_path = os.path.join(current_dir, '..', 'configs', 'da1_augmentation.yaml')   
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path, help='Path to the config file.')
    opt = parser.parse_args()

    if not os.path.exists(opt.config):
        raise FileNotFoundError(f"Config file not found: {opt.config}") 
    
    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)  
    
    run_processing(config)