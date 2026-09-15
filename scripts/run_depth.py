import os
import argparse
#import skimage
import torch
import yaml
import numpy as np
import sys

# import modules from src even if we are in scripts folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation.depth_loader import load_depth_model

def run_processing(config):
    print("COnfiguration loaded: ", config.keys())
    dataset_root = config['paths']['dataset_root']
    output_root = config['paths']['output_root']
    checkpoints_root = config['paths']['checkpoints_root']


    for model in config['depth']['model_dirs']:
        model_dir = os.path.join(checkpoints_root , model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depth_model = load_depth_model(model_dir, device)
        
        print("MODEL_DIR = ", model_dir)
        for dataset in config['datasets']['folders']:
            images_path = os.path.join(dataset_root, dataset, "images")

            print(f"dataset root = {images_path}")
            saving_depth_folder = os.path.join(output_root, "depth", model, dataset)
            print(f"Saving Depth Maps of {dataset} to: {saving_depth_folder}")

            os.makedirs(saving_depth_folder, exist_ok=True)

            file_names = os.listdir(images_path)

            count = 0
            for filename in file_names:
                tI = os.path.join(images_path, filename)

                print (f"Path immagine:c{tI}")
                # Run inference of depth 
                images = [tI]
                prediction = depth_model.inference(
                    images,
                )

                print(f"nome file: {filename} depth map shape: {prediction.depth[0].shape} dtype: {prediction.depth[0].dtype} min: {prediction.depth[0].min():.4f} max: {prediction.depth[0].max():.4f}")
                name = os.path.splitext(filename)[0]

                # Save the depth map to the specified folder
                np.save(os.path.join(saving_depth_folder, name + ".npy"), prediction.depth[0])
                
                from dataclasses import fields

                if count == 0:
                    print("Depth prediction fields:")
                    for f in fields(prediction):
                        v = getattr(prediction, f.name)
                        if isinstance(v, np.ndarray):
                            print(f"{f.name:20s} {v.shape} {v.dtype}  min={v.min():.4f} max={v.max():.4f}")
                        else:
                            print(f"{f.name:20s} {type(v).__name__}  {v}")

                count = count + 1
                print("(-) ",dataset, ": ", count,"/",len(file_names))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'configs', 'depth_augmentation.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default=config_path)
    opt = parser.parse_args()

    if not os.path.exists(opt.config):
        raise FileNotFoundError(f"COnfig file not found: {opt.config}")

    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    run_processing(config)