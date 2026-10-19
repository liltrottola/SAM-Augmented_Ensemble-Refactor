import os
import argparse
import sys
import csv
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
    parser.add_argument('--name', type=str, default=None, help='name for this ensemble run')

    args = parser.parse_args()

    opt = Config(load_config(args.config))

    # Override with command line arguments if provided
    if args.out_folder is not None:
        opt.paths.out_folder = args.out_folder
    if args.models_outputs is not None:
        opt.paths.models_outputs = args.models_outputs
    if args.test_masks is not None:
        opt.paths.test_masks = args.test_masks
    if args.name is not None:
        opt.paths.out_folder = os.path.join(opt.paths.out_folder, args.name)

    #each subfolder of --models_outputs contains the output for the whole 5 polyp datasets.
    models_to_sum = os.listdir(opt.paths.models_outputs)
    buffer = []
    dataset_names = []

    model1_path = os.path.join(opt.paths.models_outputs, models_to_sum[0])
    datasets = os.listdir(model1_path)
    for item in datasets:
        dataset_names.append(item)
        current_labels_path = os.path.join(opt.paths.test_masks, item, "masks")
        current_output_folder = os.path.join(opt.paths.out_folder, item)

        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)

        process_images_in_folder(opt.paths.models_outputs, current_output_folder, item)
        buffer.append(get_dice(current_labels_path, current_output_folder, item))
        
    #mean across all datasets
    mean_dice = sum(buffer) / len(buffer)
    print("mean", mean_dice)

    #save the models used and the dice per dataset in the ensemble folder
    ensemble_name = os.path.basename(os.path.normpath(opt.paths.out_folder))
    models = ";".join(sorted(models_to_sum))
    csv_path = os.path.join(opt.paths.out_folder, "dice.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ensemble", "models", "dataset", "dice"])
        for i in range(len(dataset_names)):
            writer.writerow([ensemble_name, models, dataset_names[i], buffer[i]])
        writer.writerow([ensemble_name, models, "mean", mean_dice])
    print("saved", csv_path)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()