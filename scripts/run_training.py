import argparse
import os
import subprocess
import sys
import yaml

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # to import from src
from src.augmentation import OFFLINE_DA_METHODS
from src.sweep.naming import build_model_name, is_valid_combo

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

def build_command(model, seed, run_id, sam_version, method, da_method, lr_method, debug):
    cmd = []
    cmd.append("python")
    cmd.append(model['train_script'])

    cmd.append("--model_name")
    model_name = build_model_name(model['name'], run_id, method, sam_version, da_method, lr_method)
    cmd.append(model_name)

    cmd.append("--seed")
    cmd.append(str(seed))

    if (sam_version is not None) and (method is not None):
        cmd.append("--sam_version")
        cmd.append(str(sam_version))
        cmd.append("--method")
        cmd.append(str(method))

    if da_method == "noda":
        # Explicit "no augmentation at all" - always pass the flag(s), never omit them.
        # Omitting the flag would leave whatever offline/online_augmentation is already
        # in the yaml config untouched, silently reusing a stale value instead of really
        # disabling DA. Train_aux.py has no --offline_da flag (aux never had an offline axis).
        cmd += ["--online_da", "noda"]
        if not model.get("has_aux", False):
            cmd += ["--offline_da", "noda"]
    elif da_method is not None:
        # offline (da1/da2) redirects image_root; online (da3) is applied at runtime by the dataloader.
        # Train_aux.py has no --offline_da flag: passing da1/da2 for an aux model fails loud (unsupported combo, not yet implemented).
        cmd.append("--offline_da" if da_method in OFFLINE_DA_METHODS else "--online_da")
        cmd.append(str(da_method))

    if lr_method is not None:
        cmd.append("--lr_method")
        cmd.append(str(lr_method))

    if debug:
        cmd.append("--debug")

    return cmd

def run_training(cmd, cwd):
    print(f"Eseguendo comando: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERRORE: Il comando è terminato con un errore (codice {e.returncode}): {' '.join(cmd)}")
        exit(1)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep", type=str, default="../configs/sweep_train.yaml", help="path to sweep config file")
    parser.add_argument("--model", type=str, default=None, help="specific model to run (overrides config file)")
    parser.add_argument("--run_id", type=int, nargs="+", default=None, help="specific run IDs to execute, e.g. --run_id 2 3 4 (overrides config file)")
    parser.add_argument("--debug", action='store_true', default=False, help="run in debug mode")
    parser.add_argument("--force", action='store_true', default=False, help="retrain even if the checkpoint already exists (default: skip it)")

    args = parser.parse_args()

    if not os.path.exists(args.sweep):
        print(f"ERRORE: FILE CONFIGURAZIONE NON TROVATO: {args.sweep}")
        exit(1)

    sweep = load_sweep_config(args.sweep)

    if args.model is None:
        models_to_run = sweep['models']
    else:
        models_to_run = [model for model in sweep['models'] if model['name'] == args.model]
        if not models_to_run:
            available_models = ", ".join(model['name'] for model in sweep['models'])
            print(f"ERRORE: MODELLO NON TROVATO NEL FILE DI CONFIGURAZIONE: {args.model}")
            print(f"Modelli disponibili: {available_models}")
            exit(1)

    if args.run_id is not None:
        run_ids = args.run_id                  # already a list thanks to nargs="+"
    else:
        run_ids = list(range(1, sweep['training']['runs'] + 1))

    for model in models_to_run:
        cwd = "../src/models/" + model['folder']
        print(f"Controllando esistenza cartella modello: {cwd}")
        
        if not os.path.exists(cwd):
            print(f"ERRORE: CARTELLA MODELLO NON TROVATA: {cwd}")
            exit(1)
        
        da_methods = sweep['training'].get('da_methods', [None])   # DA axis; without key -> 1 legacy run
        lr_methods = sweep['training'].get('lr_methods', [None])   # retrocompat: senza chiave -> 1 run legacy
        for run_id in run_ids:
            if model['has_aux']:
                for sam_version in sweep['training']['sam_versions']:
                    for method in sweep['training']['aug_methods']:
                        for da_method in da_methods:
                            # aux models only support the online DA axis (da3/null); da1/da2 here is the
                            # deferred "SAM regenerated over offline-DA data" case, not yet implemented.
                            if not is_valid_combo(model, da_method):
                                print(f"ERRORE: offline DA method '{da_method}' not supported for aux models yet (model={model['name']}) -- skipping")
                                continue
                            for lr_method in lr_methods:
                                model_name = build_model_name(model['name'], run_id, method, sam_version, da_method, lr_method)
                                if checkpoint_exists(model, model_name) and not args.force:
                                    print(f"SKIP (already trained): {model_name}")
                                    continue
                                cmd = build_command(model, sweep['training']['seeds'][run_id - 1], run_id, sam_version, method, da_method, lr_method, args.debug)
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