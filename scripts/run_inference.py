import argparse
import os
import subprocess
import sys
import yaml

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # to import from src
from src.augmentation import OFFLINE_DA_METHODS
from src.sweep.naming import build_model_name, get_aux_sources, is_valid_combo

def load_sweep_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_model_pth(model, run_id, method=None, aux_source=None, da_method=None, lr_method=None):
    base = build_model_name(model, run_id, method, aux_source, da_method, lr_method)
    return f"{base}.pth"

def build_command(model, run_id, aux_source, method, da_method, lr_method):
    cmd = []
    cmd.append("python")
    cmd.append(model['test_script'])

    cmd.append("--model_pth")
    model_pth = build_model_pth(model['name'], run_id, method, aux_source, da_method, lr_method)
    cmd.append(model_pth)

    if (aux_source is not None) and (method is not None):
        cmd.append("--aux_source")
        cmd.append(str(aux_source))
        cmd.append("--method")
        cmd.append(str(method))

    return cmd

def run_inference(cmd, cwd):
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: the command failed (exit code {e.returncode}): {' '.join(cmd)}")
        exit(1)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep", type=str, default="../configs/sweep_test.yaml", help="path to sweep config file")
    parser.add_argument("--model", type=str, default=None, help="specific model to run (overrides config file)")
    parser.add_argument("--run_id", type=int, nargs="+", default=None, help="specific run IDs to execute, e.g. --run_id 2 3 4 (overrides config file)")
    parser.add_argument("--aux_source", type=str, nargs="+", default=None, help="specific aux sources to run for the aux models, e.g. --aux_source sam2_stab085 da3mono-large (must be keys of aux_sources in the sweep config; default: all)")

    args = parser.parse_args()

    if not os.path.exists(args.sweep):
        print(f"ERROR: CONFIG FILE NOT FOUND: {args.sweep}")
        exit(1)

    sweep = load_sweep_config(args.sweep)

    if args.model is None:
        models_to_run = sweep['models']
    else:
        models_to_run = [model for model in sweep['models'] if model['name'] == args.model]
        if not models_to_run:
            available_models = ", ".join(model['name'] for model in sweep['models'])
            print(f"ERROR: MODEL NOT FOUND IN THE CONFIG FILE: {args.model}")
            print(f"Available models: {available_models}")
            exit(1)

    if args.run_id is not None:
        run_ids = args.run_id                  # already a list thanks to nargs="+"
    else:
        run_ids = list(range(1, sweep['testing']['runs'] + 1))

    # (aux_source, method) pairs for the aux models, optionally restricted to the --aux_source given on the CLI
    aux_pairs = get_aux_sources(sweep['testing'])
    if args.aux_source is not None:
        available_sources = list(sweep['testing']['aux_sources'])
        unknown = [s for s in args.aux_source if s not in available_sources]
        if unknown:
            print(f"ERROR: AUX SOURCE NOT FOUND IN THE CONFIG FILE: {unknown}")
            print(f"Available aux sources: {', '.join(available_sources)}")
            exit(1)
        aux_pairs = [(source, method) for source, method in aux_pairs if source in args.aux_source]

    for model in models_to_run:
        cwd = "../src/models/" + model['folder']
        print(f"Checking model folder exists: {cwd}")
        
        if not os.path.exists(cwd):
            print(f"ERROR: MODEL FOLDER NOT FOUND: {cwd}")
            exit(1)
        
        da_methods = sweep['testing'].get('da_methods', [None])   # DA axis; without key -> 1 legacy run
        lr_methods = sweep['testing'].get('lr_methods', [None])   # backward compat: without key -> 1 legacy run
        for run_id in run_ids:
            if model['has_aux']:
                for aux_source, method in aux_pairs:
                    for da_method in da_methods:
                        # mirror run_training.py's aux guard, or the checkpoint filename won't match
                        if not is_valid_combo(model, da_method):
                            print(f"ERROR: offline DA method '{da_method}' not supported for aux models yet (model={model['name']}) -- skipping")
                            continue
                        for lr_method in lr_methods:
                            cmd = build_command(model, run_id, aux_source, method, da_method, lr_method)
                            run_inference(cmd, cwd)
            else:
                for da_method in da_methods:
                    for lr_method in lr_methods:
                        cmd = build_command(model, run_id, None, None, da_method, lr_method)
                        run_inference(cmd, cwd)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
