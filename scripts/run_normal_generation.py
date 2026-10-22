import os
import sys
import argparse
import numpy as np
import torch
import yaml
from PIL import Image

# import modules from src even if we are in scripts folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation.normal_loader import load_dsine_model
from src.augmentation import normal_methods


def run_processing(config):
    print("Configuration loaded: ", config.keys())
    dataset_root = config['paths']['dataset_root']
    output_root = config['paths']['output_root']
    dsine_root = config['normals']['dsine_root']
    config_txt = config['normals']['config_txt']
    fov = config['normals'].get('fov', 60.0)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available: normal generation must run on GPU. "
            "Check that venv_dsine is active"
        )
    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    model = load_dsine_model(dsine_root, config_txt, device)

    for dataset in config['datasets']['folders']:
        images_path = os.path.join(dataset_root, dataset, "images")
        print(f"dataset root = {images_path}")

        saving_folder = os.path.join(output_root, "normals", "dsine", dataset)
        print(f"Saving Normal Maps of {dataset} to: {saving_folder}")

        # raw normal maps in .npy format (float32, [-1, 1])
        raw_folder = os.path.join(saving_folder, "npy")
        os.makedirs(raw_folder, exist_ok=True)

        # visualizations (uint8 png, only for visual inspection)
        vis_folder = os.path.join(saving_folder, "vis")
        os.makedirs(vis_folder, exist_ok=True)

        file_names = sorted(os.listdir(images_path))

        count = 0
        for filename in file_names:
            tI = os.path.join(images_path, filename)
            name = os.path.splitext(filename)[0]

            normals = normal_methods.raw_normal_generation(model, tI, device, fov)
            np.save(os.path.join(raw_folder, name + ".npy"), normals)

            vis = (((normals + 1) * 0.5) * 255).astype(np.uint8)
            Image.fromarray(vis).save(os.path.join(vis_folder, name + ".png"))

            count = count + 1
            print("(-) ", dataset, ": ", count, "/", len(file_names))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(current_dir, '../configs/normal_generation.yaml'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path)
    opt = parser.parse_args()

    if not os.path.exists(opt.config):
        raise FileNotFoundError(f"Config file not found: {opt.config}")

    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    run_processing(config)
