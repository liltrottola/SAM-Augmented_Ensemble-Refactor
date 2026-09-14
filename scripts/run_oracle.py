import os
import argparse
import sys
import yaml
import importlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from src.ensemble.oracle1 import run_oracle

#HARDCODED DICTIONARY OF CRITICAL IMAGES TO MONITOR (Images where OracleW assigned a zero weight to specific experts within the huge(H) ensemble, sorted in descending order by the number of zeroed experts)
CRITICAL_IMAGES = {
        "CVC-ColonDB": ["342.png", "325.png", "357.png", "360.png", "380.png", "324.png", "352.png", "358.png", "353.png", "361.png", "71.png", "379.png", "341.png", "278.png", "298.png", "116.png", "267.png", "301.png", "115.png", "319.png", "333.png", "345.png", "346.png", "351.png", "356.png", "378.png", "350.png", "359.png", "375.png", "2.png", "30.png", "322.png","349.png", "354.png", "355.png", "377.png", "154.png", "292.png", "314.png", "323.png", "343.png", "374.png", "376.png"],
        "ETIS-LaribPolypDB": ["195.png", "191.png", "192.png", "36.png", "193.png", "156.png", "60.png", "196.png", "23.png", "194.png", "152.png", "57.png", "12.png", "8.png", "161.png", "19.png", "99.png", "10.png", "101.png", "102.png", "104.png", "70.png", "74.png", "79.png", "100.png", "103.png", "71.png", "11.png", "20.png", "22.png", "24.png", "31.png", "53.png", "72.png", "78.png", "89.png", "9.png"]
        }

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/ensemble.yaml')
    parser.add_argument('--out_folder', type=str, default=None)
    parser.add_argument('--models_outputs', type=str, default=None)
    parser.add_argument('--test_masks', type=str, default=None)
    parser.add_argument('--mode', type=str, default='oracle1', help='Nome del file in src/ensemble senza .py')
    args = parser.parse_args()

    opt = Config(load_config(args.config))

    oracle_module = importlib.import_module(f"src.ensemble.{args.mode}")
    run_func = oracle_module.run_oracle

    # Paths override
    if args.out_folder: opt.paths.out_folder = args.out_folder
    if args.models_outputs: opt.paths.models_outputs = args.models_outputs
    if args.test_masks: opt.paths.test_masks = args.test_masks

    models_to_sum = os.listdir(opt.paths.models_outputs)
    buffer = []
    
    # Dictionary to store critical results across all datasets to print at the end
    all_critical_results = {}

    model1_path = os.path.join(opt.paths.models_outputs, models_to_sum[0])
    datasets = os.listdir(model1_path)

    for item in datasets:
        current_labels_path = os.path.join(opt.paths.test_masks, item)
        current_output_folder = os.path.join(opt.paths.out_folder, item, "oracle")

        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)
        
        # Execute the specific oracle logic
        result = run_func(opt.paths.models_outputs, current_labels_path, item)
        
        # Checks if run_func returns (mDice, image_scores) or just mDice
        if isinstance(result, tuple) and len(result) == 2:
            mDice, image_scores = result
            
            # Extract critical scores for the current dataset
            current_targets = CRITICAL_IMAGES.get(item, [])
            if current_targets:
                dataset_critical_scores = {}
                for img in current_targets:
                    if img in image_scores:
                        dataset_critical_scores[img] = image_scores[img]
                if dataset_critical_scores:
                    all_critical_results[item] = dataset_critical_scores
        else:
            mDice = result
            
        buffer.append(mDice)
        print(f"{item} mDICE: {mDice:.3f}")

    print(f"mean {sum(buffer) / len(buffer):.3f}")
    
    if all_critical_results:
        print(f"\nCritical images results ({args.mode}):")

        global_critical_dices = []
        for dataset_name, scores_dict in all_critical_results.items():
            print(f"\nDataset: {dataset_name}")
            for img_name, dice in scores_dict.items():
                print(f"{img_name}: {dice:.3f}")
                global_critical_dices.append(dice)

        if global_critical_dices:
            mDice_critical_total = sum(global_critical_dices) / len(global_critical_dices)
            print(f"\nTotal mDice (Critical Images Only): {mDice_critical_total:.3f}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
