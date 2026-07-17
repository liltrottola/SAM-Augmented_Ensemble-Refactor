import os
import sys
import yaml
import skimage
from skimage import io
import numpy as np
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation import da2_methods
from src.augmentation import da1_methods   # riuso has_enough_foreground

# Stain normalization methods: require a target image
STAIN_METHODS = {'rgb_histogram_match', 'reinhard_normalize', 'macenko_normalize'}


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

    target_pool_size = config['augmentation'].get('target_pool_size', 90)
    random_seed = config['augmentation'].get('random_seed', 42)
    np.random.seed(random_seed)                     # reproducibility of the generated dataset

    for cur_dir in config['datasets']['folders']:
        image_folder = os.path.join(dataset_root, cur_dir, "images")
        mask_folder = os.path.join(dataset_root, cur_dir, "masks")

        saving_path_images = os.path.join(output_root, "da2", source_name, cur_dir, "images")
        saving_path_masks = os.path.join(output_root, "da2", source_name, cur_dir, "masks")

        print(f"Processing dataset folder: {image_folder}")
        print(f"Saving augmented images to: {saving_path_images}")
        print(f"Saving augmented masks to: {saving_path_masks}")   

        os.makedirs(saving_path_images, exist_ok=True)
        os.makedirs(saving_path_masks, exist_ok=True)

        file_names = sorted(os.listdir(image_folder))  #sorted for reproducibility of the target pool selection

        # target pool for stain norm: first N images
        pool = file_names[:min(target_pool_size, len(file_names))]
        count = 0

        for filename in file_names:
            image_path = os.path.join(image_folder, filename)
            mask_path = os.path.join(mask_folder, filename)

            tI = io.imread(image_path)
            tM = io.imread(mask_path)

            if tM.ndim == 3:
                tM = tM.max(axis=-1)    # 3D RGB is triple → 2D (max per pixel) DA2 methods assumes 2D masks

            # save original
            if keep_original:
                io.imsave(os.path.join(saving_path_images, filename), tI)
                io.imsave(os.path.join(saving_path_masks, filename), tM)
                count += 1

            for method_name in method_list:
                if not hasattr(da2_methods, method_name):
                    print(f"Method '{method_name}' not in da2_methods.py — skipping")
                    continue
                aug_function = getattr(da2_methods, method_name)

                try:
                    if method_name in STAIN_METHODS:
                        # random target from the pool (replicates MATLAB j=round(1+rand*89))
                        target_name = pool[np.random.randint(len(pool))]
                        target = io.imread(os.path.join(image_folder, target_name))
                        augmented_image, augmented_mask = aug_function(tI, tM, target)
                    else:
                        augmented_image, augmented_mask = aug_function(tI, tM)

                    # foreground filter (relevant for geometric methods that move the mask)
                    if filter_enabled and not da1_methods.has_enough_foreground(augmented_mask, min_pixels):
                        print(f"Skip {filename} [{method_name}]: <{min_pixels} foreground px")
                        continue

                    aug_filename = f"{os.path.splitext(filename)[0]}_{method_name}{os.path.splitext(filename)[1]}"
                    io.imsave(os.path.join(saving_path_images, aug_filename), augmented_image)
                    io.imsave(os.path.join(saving_path_masks, aug_filename), augmented_mask)
                    count += 1

                except Exception as e:
                    print(f"[{method_name}] failed on {filename}: {e} — skipping")
                    continue

        print(f"Total augmented images saved for {cur_dir}: {count}")


if __name__ == "__main__":
    # Find directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to config file
    config_path = os.path.join(current_dir, '..', 'configs', 'da2_augmentation.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path, help='Path to the config file.')
    opt = parser.parse_args()

    if not os.path.exists(opt.config):
        raise FileNotFoundError(f"Config file not found: {opt.config}")

    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    run_processing(config)
