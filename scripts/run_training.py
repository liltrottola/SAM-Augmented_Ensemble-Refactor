import argparse
import os
import subprocess
import sys
import yaml

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # to import from src
from src.augmentation import OFFLINE_DA_METHODS
from src.sweep.naming import build_model_name, get_aux_sources

def load_sweep_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def checkpoint_exists(model, model_name):
    """True if the checkpoint already exists, using models_dir from the model's yaml.

    models_dir is relative to the model folder (src/models/<folder>), which is
    where Train_*.py runs; we join it with that folder to resolve from scripts/.
    """
    cfg = load_sweep_config(model['config'])
    models_dir = cfg['paths']['models_dir']                       # e.g. "../../../output/models/"
    model_folder = os.path.join("..", "src", "models", model['folder'])
    path = os.path.join(model_folder, models_dir, f"{model_name}.pth")
    return os.path.exists(path)

def build_command(model, seed, run_id, aux_source, method, da_method, lr_method, debug):
    cmd = []
    cmd.append("python")
    cmd.append(model['train_script'])

    cmd.append("--model_name")
    model_name = build_model_name(model['name'], run_id, method, aux_source, da_method, lr_method)
    cmd.append(model_name)

    cmd.append("--seed")
    cmd.append(str(seed))

    if (aux_source is not None) and (method is not None):
        cmd.append("--aux_source")
        cmd.append(str(aux_source))
        cmd.append("--method")
        cmd.append(str(method))

    if da_method is not None:
        # Always pass BOTH online and offline axes of augmentation explicitly, 
        # never just the chosen one: leaving the other axis unmentioned falls back 
        # to whatever offline_augmentation/online_augmentation is already in the yaml config, 
        # silently stacking it on top instead of really isolating the chosen DA method.
        offline_aug = da_method if da_method in OFFLINE_DA_METHODS else "noda"
        online_aug = da_method if da_method == "da3" else "noda"
        cmd += ["--online_da", online_aug]
        cmd += ["--offline_da", offline_aug]

    if lr_method is not None:
        cmd.append("--lr_method")
        cmd.append(str(lr_method))

    if debug:
        cmd.append("--debug")

    return cmd

def run_training(cmd, cwd):
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: the command failed (exit code {e.returncode}): {' '.join(cmd)}")
        exit(1)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep", type=str, default="../configs/sweep_train.yaml", help="path to sweep config file")
    parser.add_argument("--model", type=str, default=None, help="specific model to run (overrides config file)")
    parser.add_argument("--run_id", type=int, nargs="+", default=None, help="specific run IDs to execute, e.g. --run_id 2 3 4 (overrides config file)")
    parser.add_argument("--debug", action='store_true', default=False, help="run in debug mode")
    parser.add_argument("--force", action='store_true', default=False, help="retrain even if the checkpoint already exists (default: skip it)")
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
        run_ids = list(range(1, sweep['training']['runs'] + 1))

    # (aux_source, method) pairs for the aux models, optionally restricted to the --aux_source given on the CLI
    aux_pairs = get_aux_sources(sweep['training'])
    if args.aux_source is not None:
        available_sources = list(sweep['training']['aux_sources'])
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
        
        da_methods = sweep['training'].get('da_methods', [None])   # DA axis; without key -> 1 legacy run
        lr_methods = sweep['training'].get('lr_methods', [None])   # backward compat: without key -> 1 legacy run
        for run_id in run_ids:
            if model['has_aux']:
                for aux_source, method in aux_pairs:
                    for da_method in da_methods:
                        # offline DA (da1/da2/clahe): Train_aux.py reads the DA train set and the aux generated on it
                        for lr_method in lr_methods:
                            model_name = build_model_name(model['name'], run_id, method, aux_source, da_method, lr_method)
                            if checkpoint_exists(model, model_name) and not args.force:
                                print(f"SKIP (already trained): {model_name}")
                                continue
                            cmd = build_command(model, sweep['training']['seeds'][run_id - 1], run_id, aux_source, method, da_method, lr_method, args.debug)
                            run_training(cmd, cwd)
            else:
                for da_method in da_methods:
                    for lr_method in lr_methods:
                        model_name = build_model_name(model['name'], run_id, None, None, da_method, lr_method)
                        if checkpoint_exists(model, model_name) and not args.force:
                            print(f"SKIP (already trained): {model_name}")
                            continue
                        cmd = build_command(model, sweep['training']['seeds'][run_id - 1], run_id, None, None, da_method, lr_method, args.debug)
                        run_training(cmd, cwd)
        
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()