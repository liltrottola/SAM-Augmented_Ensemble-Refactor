import os
import sys
import yaml
import argparse
import numpy as np
import skimage
from skimage import io

# Add the parent directory to the sys.path to allow imports from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation import light_methods

# For every training image writes the original (if keep_original) and the light variants:
#   <name>_rl1 ... <name>_rlN       relighting, one per light direction   (needs the DSINE normals)
#   <name>_att1 ... <name>_attN     depth-guided attenuation, one per floor (needs the DA3 depth)
#   <name>_clahe1 ... <name>_claheN CLAHE-LAB, one per clip limit
# The mask is copied unchanged for every variant (photometric transforms only).


def read_rgb(path):
    """grayscale -> 3 channels, RGBA -> RGB, uint8."""
    tI = io.imread(path)
    if tI.ndim == 2:
        tI = np.stack([tI] * 3, axis=-1)
    if tI.shape[2] == 4:
        tI = tI[:, :, :3]
    if tI.dtype != np.uint8:
        tI = skimage.img_as_ubyte(tI)
    return tI


def run_processing(config):
    print("Configuration loaded:", config.keys())
    dataset_root = config['paths']['dataset_root']
    output_root = config['paths']['output_root']
    depth_root = config['paths']['depth_root']
    normal_root = config['paths']['normal_root']

    keep_original = config['augmentation'].get('keep_original', True)
    light_directions = config['augmentation'].get('light_directions') or []
    floors = config['augmentation'].get('floors') or []
    clip_limits = config['augmentation'].get('clip_limits') or []

    for cur_dir in config['datasets']['folders']:
        #Path input
        image_folder = os.path.join(dataset_root, cur_dir, "images")
        mask_folder = os.path.join(dataset_root, cur_dir, "masks")
        depth_folder = os.path.join(depth_root, cur_dir, "npy")
        normal_folder = os.path.join(normal_root, cur_dir, "npy")

        #Path output
        saving_path_images = os.path.join(output_root, "light", cur_dir, "images")
        saving_path_masks = os.path.join(output_root, "light", cur_dir, "masks")

        print(f"Processing dataset folder: {image_folder}")
        print(f"Depth from: {depth_folder}")
        print(f"Normals from: {normal_folder}")
        print(f"Saving augmented images to: {saving_path_images}")
        print(f"Saving augmented masks to: {saving_path_masks}")
        os.makedirs(saving_path_images, exist_ok=True)
        os.makedirs(saving_path_masks, exist_ok=True)

        file_names = sorted(os.listdir(image_folder))

        n_images = 0
        n_written = 0
        n_skipped = 0
        missing_depth = []
        missing_normals = []

        for filename in file_names:
            name, ext = os.path.splitext(filename)
            image_path = os.path.join(image_folder, filename)
            mask_path = os.path.join(mask_folder, filename)

            tI = read_rgb(image_path)
            tM = io.imread(mask_path)

            if tM.ndim == 3:
                tM = tM.max(axis=-1)    # 3D RGB is triple → 2D (max per pixel)

            n_images += 1

            # Save the original if requested
            if keep_original:
                io.imsave(os.path.join(saving_path_images, filename), tI)
                io.imsave(os.path.join(saving_path_masks, filename), tM)
                n_written += 1

            # Relighting: needs the DSINE normals
            if len(light_directions) > 0:
                normal_path = os.path.join(normal_folder, name + ".npy")
                if os.path.exists(normal_path):
                    normals = np.load(normal_path)
                    for i, light in enumerate(light_directions, start=1):
                        aug_image = light_methods.relight(tI, normals, light)
                        aug_filename = f"{name}_rl{i}{ext}"
                        io.imsave(os.path.join(saving_path_images, aug_filename), aug_image)
                        io.imsave(os.path.join(saving_path_masks, aug_filename), tM)
                        n_written += 1
                else:
                    print(f"WARNING: missing normals for {name} ({normal_path}), skipping the rl variants.")
                    missing_normals.append(name)
                    n_skipped += len(light_directions)

            # Attenuation: needs the DA3 depth
            if len(floors) > 0:
                depth_path = os.path.join(depth_folder, name + ".npy")
                if os.path.exists(depth_path):
                    depth = np.load(depth_path)
                    for i, floor in enumerate(floors, start=1):
                        aug_image = light_methods.attenuate(tI, depth, floor)
                        aug_filename = f"{name}_att{i}{ext}"
                        io.imsave(os.path.join(saving_path_images, aug_filename), aug_image)
                        io.imsave(os.path.join(saving_path_masks, aug_filename), tM)
                        n_written += 1
                else:
                    print(f"WARNING: missing depth for {name} ({depth_path}), skipping the att variants.")
                    missing_depth.append(name)
                    n_skipped += len(floors)

            # CLAHE: always available
            for i, clip_limit in enumerate(clip_limits, start=1):
                aug_image = light_methods.clahe(tI, clip_limit)
                aug_filename = f"{name}_clahe{i}{ext}"
                io.imsave(os.path.join(saving_path_images, aug_filename), aug_image)
                io.imsave(os.path.join(saving_path_masks, aug_filename), tM)
                n_written += 1

        print(f"Summary for {cur_dir}:")
        print(f"  images processed: {n_images}")
        print(f"  images written (originals included): {n_written}")
        print(f"  variants skipped: {n_skipped}")
        print(f"  images without normals: {len(missing_normals)}")
        print(f"  images without depth: {len(missing_depth)}")


if __name__ == "__main__":

    # Find directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to config file
    config_path = os.path.join(current_dir, '..', 'configs', 'light_augmentation.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path, help='Path to the config file.')
    opt = parser.parse_args()

    if not os.path.exists(opt.config):
        raise FileNotFoundError(f"Config file not found: {opt.config}")

    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    run_processing(config)
