import os
import argparse
import torch
import yaml
import numpy as np
import sys

# import modules from src even if we are in scripts folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation.depth_loader import load_depth_model
from src.augmentation import depth_methods

def run_processing(config):
    print("Configuration loaded: ", config.keys())
    dataset_root = config['paths']['dataset_root']
    output_root = config['paths']['output_root']
    checkpoints_root = config['paths']['checkpoints_root']


    for model in config['depth']['model_dirs']:
        model_dir = os.path.join(checkpoints_root , model)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available: depth generation must run on GPU. "
                "Check that venv_dav3 is active"
            )
        device = torch.device("cuda")
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

        depth_model = load_depth_model(model_dir, device)
        process_res = config['depth']['process_res']
        process_res_method = config['depth']['process_res_method']

        print("MODEL_DIR = ", model_dir)
        for dataset in config['datasets']['folders']:
            images_path = os.path.join(dataset_root, dataset, "images")

            print(f"dataset root = {images_path}")
            saving_depth_folder = os.path.join(output_root, "depth", model, dataset)
            print(f"Saving Depth Maps of {dataset} to: {saving_depth_folder}")

            # Create a folder for raw depth maps in .npy format
            raw_folder = os.path.join(saving_depth_folder, "npy")
            os.makedirs(raw_folder, exist_ok=True)

            # Create a folder for visualizations of the depth maps
            vis_folder = os.path.join(saving_depth_folder, "vis")
            os.makedirs(vis_folder, exist_ok=True)

            file_names = os.listdir(images_path)

            count = 0
            for filename in file_names:
                tI = os.path.join(images_path, filename)
                name = os.path.splitext(filename)[0]
                
                # Run inference of depth 
                depth = depth_methods.raw_depth_generation(depth_model, tI, process_res, process_res_method)
                # Save the depth map to the specified folder
                np.save(os.path.join(raw_folder, name + ".npy"), depth)

                
                count = count + 1
                print("(-) ",dataset, ": ", count,"/",len(file_names))


if __name__ == "__main__":
    # Find directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # build the config file path    
    config_path = os.path.abspath(os.path.join(current_dir, '../configs/depth_generation.yaml'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default=config_path)
    opt = parser.parse_args()

    if not os.path.exists(opt.config):
        raise FileNotFoundError(f"Config file not found: {opt.config}")

    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    run_processing(config)