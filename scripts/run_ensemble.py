import os
import argparse
import sys
import yaml

# Per importare i moduli da 'src' anche se siamo in 'scripts'
# Aggiunge la cartella superiore al path di Python
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ensemble.ensemble import get_dice
from src.ensemble.ensemble import process_images_in_folder

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

def main():
    #get user input
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/ensemble.yaml', help='path to ensemble config file')
    parser.add_argument('--out_folder', type=str, default=None)
    parser.add_argument('--models_outputs', type=str, default=None)
    parser.add_argument('--test_masks', type=str, default=None)
    parser.add_argument('--ensemble_name', type=str, default=None, help='name for this ensemble run')

    args = parser.parse_args()

    opt = Config(load_config(args.config))

    # Override with command line arguments if provided
    if args.out_folder is not None:
        opt.paths.out_folder = args.out_folder
    if args.models_outputs is not None:
        opt.paths.models_outputs = args.models_outputs
    if args.test_masks is not None:
        opt.paths.test_masks = args.test_masks
    if args.ensemble_name is not None:
        opt.paths.out_folder = os.path.join(opt.paths.out_folder, args.ensemble_name)

    #each subfolder of --models_outputs contains the output for the whole 5 polyp datasets.
    models_to_sum = os.listdir(opt.paths.models_outputs)
    buffer = []

    model1_path = os.path.join(opt.paths.models_outputs, models_to_sum[0])
    datasets = os.listdir(model1_path)
    for item in datasets:
        current_labels_path = os.path.join(opt.paths.test_masks, item, "masks")
        current_output_folder = os.path.join(opt.paths.out_folder, item, "mean")

        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)

        process_images_in_folder(opt.paths.models_outputs, current_output_folder, item)
        buffer.append(get_dice(current_labels_path, current_output_folder, item))
        
    #mean across all datasets
    print("mean", sum(buffer) / len(buffer))

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()