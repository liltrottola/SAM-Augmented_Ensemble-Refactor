import argparse
import os
import subprocess
import sys
import yaml

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # to import from src
from src.augmentation import OFFLINE_DA_METHODS, ONLINE_DA_METHODS, NO_DA
from src.sweep.naming import build_model_name
from src.sweep import registry as reg   # sweep axes: CLI value or sweep yaml, names checked against the registry / code

def load_sweep_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def checkpoint_exists(registry, model, model_name):
    """True if the checkpoint already exists, using models_dir from the model's yaml.

    models_dir is relative to the model folder (<models_root>/<folder>, models_root in the registry), which is
    where Train_*.py runs; we join it with that folder to resolve from scripts/.
    """
    cfg = load_sweep_config(model['config'])
    models_dir = cfg['paths']['models_dir']                       # e.g. "../../../output/models/"
    model_folder = os.path.join(registry['models_root'], model['folder'])
    path = os.path.join(model_folder, models_dir, f"{model_name}.pth")
    return os.path.exists(path)

def build_command(model, seed, run_id, aux_source, method, da_method, lr_method, debug, priors=None, prior_flags=None, train_dataset=None):
    cmd = []
    cmd.append("python")
    cmd.append(model['train_script'])

    cmd.append("--model_name")
    model_name = build_model_name(model['name'], run_id, method, aux_source, da_method, lr_method, priors, train_dataset)
    cmd.append(model_name)

    if train_dataset is not None:
        # train set (folder under datasets/): replaces datasets.train of the model yaml
        cmd.append("--train_dataset")
        cmd.append(str(train_dataset))

    cmd.append("--seed")
    cmd.append(str(seed))

    if (aux_source is not None) and (method is not None):
        cmd.append("--aux_source")
        cmd.append(str(aux_source))
        cmd.append("--method")
        cmd.append(str(method))

    if prior_flags is not None:
        # N3 models: the prior channels, one flag per family (--sam_priors ... --depth_priors ...), the same passed to the test
        cmd += prior_flags

    if da_method is not None:
        # Always pass BOTH online and offline axes of augmentation explicitly, 
        # never just the chosen one: leaving the other axis unmentioned falls back 
        # to whatever offline_augmentation/online_augmentation is already in the yaml config, 
        # silently stacking it on top instead of really isolating the chosen DA method.
        offline_aug = da_method if da_method in OFFLINE_DA_METHODS else NO_DA
        online_aug = da_method if da_method in ONLINE_DA_METHODS else NO_DA
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
    parser.add_argument("--model", type=str, default=None, help="models to run (overrides config file)")
    parser.add_argument("--run_id", type=int, nargs="+", default=None, help="specific run IDs to execute, e.g. --run_id 2 3 4 (overrides config file)")
    parser.add_argument("--debug", action='store_true', default=False, help="run in debug mode")
    parser.add_argument("--force", action='store_true', default=False, help="retrain even if the checkpoint already exists (default: skip it)")
    parser.add_argument("--da_method", type=str, nargs="+", default=None, help="DA methods (default: da_methods of the sweep)")
    parser.add_argument("--train_dataset", type=str, nargs="+", default=None, help="train sets, folders under datasets/ (default: train_datasets of the sweep)")
    
    parser.add_argument("--aux_source", type=str, default=None, help="specific aux source to run for the aux models, e.g. --aux_source  da3mono-large (default: all aux_sources of the sweep)")
    parser.add_argument("--lr_method", type=str, nargs="+", default=None, help="specific LR methods to run, e.g. --lr_method lra lrb ( default: all lr_methods of the sweep)")
    parser.add_argument("--aux_methods", type=str, nargs="+", default=None, help="methods of --aux source to run(need --aux_source) (default: all methods of the sweep)")
    # N3 models: prior channels per family (families and folders in prior_sources of the registry); if at least one is
    # given they replace prior_sources of the sweep, e.g. --sam_priors segPrior boundaryPrior --depth_priors frst_prior
    parser.add_argument("--sam_priors", type=str, nargs="+", default=None, help="N3 models: SAM priors to use as channels (default: prior_sources of the sweep)")
    parser.add_argument("--depth_priors", type=str, nargs="+", default=None, help="N3 models: depth priors to use as channels (default: prior_sources of the sweep)")
    parser.add_argument("--dsine_priors", type=str, nargs="+", default=None, help="N3 models: DSINE priors to use as channels (default: prior_sources of the sweep)")
    parser.add_argument("--registry", type=str, default="../configs/sweep_registry.yaml", help="path to registry file")

    args = parser.parse_args()

    for path in (args.sweep, args.registry):
        if not os.path.exists(path):
            print(f"ERROR: CONFIG FILE NOT FOUND: {path}")
            exit(1)

    sweep = load_sweep_config(args.sweep)
    sweep_train_section = sweep['training']
    registry = reg.load_registry(args.registry)

    # every axis: the CLI value if given, otherwise the sweep one; names checked against the registry / the code
    models_to_run = reg.get_models(registry, sweep['models'], args.model)
    aux_pairs = reg.get_aux_pairs(registry, sweep_train_section, args.aux_source, args.aux_methods)
    da_methods = reg.get_da_methods(sweep_train_section, args.da_method)
    lr_methods = reg.get_lr_methods(sweep_train_section, args.lr_method)
    run_ids = reg.get_run_ids(sweep_train_section, args.run_id)
    train_datasets = reg.get_train_datasets(sweep_train_section, args.train_dataset)

    # N3 prior channels: computed (and checked) only if an N3 model is launched, so the other models do not need
    # prior_sources in the sweep
    n3_priors = None
    n3_prior_flags = None
    for model in models_to_run:
        if model.get('n3'):
            cli_priors = {"sam": args.sam_priors, "depth": args.depth_priors, "dsine": args.dsine_priors}
            n3_priors = reg.get_n3_priors(registry, sweep_train_section, cli_priors)   # checked, ordered, for the name
            print(f"N3 prior channels: {n3_priors}")

            # N3: the prior flags passed to Train_N3, the same given to this runner (or prior_sources of the sweep)
            if args.sam_priors or args.depth_priors or args.dsine_priors:
                chosen = cli_priors
            else:
                chosen = sweep_train_section.get('prior_sources') or {}
            n3_prior_flags = []
            for family, priors in chosen.items():
                if priors:
                    n3_prior_flags.append(f"--{family}_priors")
                    n3_prior_flags += priors
            break

    for model in models_to_run:
        cwd = os.path.join(registry['models_root'], model['folder'])
        print(f"Checking model folder exists: {cwd}")
        
        if not os.path.exists(cwd):
            print(f"ERROR: MODEL FOLDER NOT FOUND: {cwd}")
            exit(1)

        for train_dataset in train_datasets:
            for run_id in run_ids:
                if model['has_aux']:
                    for aux_source, method in aux_pairs:
                        for da_method in da_methods:
                            # offline DA (da1/da2/clahe): Train_aux.py reads the DA train set and the aux generated on it
                            for lr_method in lr_methods:
                                model_name = build_model_name(model['name'], run_id, method, aux_source, da_method, lr_method, None, train_dataset)
                                if checkpoint_exists(registry, model, model_name) and not args.force:
                                    print(f"SKIP (already trained): {model_name}")
                                    continue
                                cmd = build_command(model, sweep_train_section['seeds'][run_id - 1], run_id, aux_source, method, da_method, lr_method, args.debug, train_dataset=train_dataset)
                                run_training(cmd, cwd)
                elif model.get('n3'):
                    # N3: as the vanilla models + the prior flags (--sam_priors ...) and the prior names in the checkpoint name;
                    # offline DA: Train_N3.py reads the DA train set and the priors generated on it
                    for da_method in da_methods:
                        for lr_method in lr_methods:
                            model_name = build_model_name(model['name'], run_id, None, None, da_method, lr_method, n3_priors, train_dataset)
                            if checkpoint_exists(registry, model, model_name) and not args.force:
                                print(f"SKIP (already trained): {model_name}")
                                continue
                            cmd = build_command(model, sweep_train_section['seeds'][run_id - 1], run_id, None, None, da_method, lr_method, args.debug, n3_priors, n3_prior_flags, train_dataset)
                            run_training(cmd, cwd)
                else:
                    for da_method in da_methods:
                        for lr_method in lr_methods:
                            model_name = build_model_name(model['name'], run_id, None, None, da_method, lr_method, None, train_dataset)
                            if checkpoint_exists(registry, model, model_name) and not args.force:
                                print(f"SKIP (already trained): {model_name}")
                                continue
                            cmd = build_command(model, sweep_train_section['seeds'][run_id - 1], run_id, None, None, da_method, lr_method, args.debug, train_dataset=train_dataset)
                            run_training(cmd, cwd)
        
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()