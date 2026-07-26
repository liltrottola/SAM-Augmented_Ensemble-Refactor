import os
import sys
import yaml
import argparse
from skimage import io
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation.clahe_lab import clahe_lab


def run_processing(config):
    print("Configurazione caricata:", config.keys())
    dataset_root = config['paths']['dataset_root']
    output_root  = config['paths']['output_root']
    source_name  = config['paths']['source_name']

    for cur_dir in config['datasets']['folders']:
        image_folder = os.path.join(dataset_root, cur_dir, "images")
        mask_folder  = os.path.join(dataset_root, cur_dir, "masks")

        saving_path_images = os.path.join(output_root, "clahe", source_name, cur_dir, "images")
        saving_path_masks  = os.path.join(output_root, "clahe", source_name, cur_dir, "masks")

        os.makedirs(saving_path_images, exist_ok=True)
        os.makedirs(saving_path_masks,  exist_ok=True)

        print(f"Processing: {image_folder}")
        print(f"Saving to:  {saving_path_images}")

        file_names = sorted(os.listdir(image_folder))
        count = 0

        for filename in file_names:
            image_path = os.path.join(image_folder, filename)
            mask_path  = os.path.join(mask_folder,  filename)

            tI = io.imread(image_path)
            tM = io.imread(mask_path)

            aug_image, aug_mask = clahe_lab(tI, tM)

            io.imsave(os.path.join(saving_path_images, filename), aug_image)
            io.imsave(os.path.join(saving_path_masks,  filename), aug_mask)
            count += 1

        print(f"Saved {count} images for {cur_dir}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'configs', 'clahe_augmentation.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path)
    opt = parser.parse_args()

    if not os.path.exists(opt.config):
        raise FileNotFoundError(f"Config file not found: {opt.config}")

    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    run_processing(config)
