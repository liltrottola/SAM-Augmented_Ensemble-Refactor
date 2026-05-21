import os
import argparse
import sys
import yaml
import importlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from src.ensemble.oracle1 import run_oracle

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

    model1_path = os.path.join(opt.paths.models_outputs, models_to_sum[0])
    datasets = os.listdir(model1_path)

    for item in datasets:
        current_labels_path = os.path.join(opt.paths.test_masks, item) # run_oracle aggiungerà /masks/
        current_output_folder = os.path.join(opt.paths.out_folder, item, "oracle")

        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)

        mDice = run_func(opt.paths.models_outputs, current_labels_path, item)
        buffer.append(mDice)

    print(f"mean {sum(buffer) / len(buffer):.3f}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
