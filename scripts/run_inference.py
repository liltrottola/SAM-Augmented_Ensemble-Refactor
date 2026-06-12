import argparse
import os
import subprocess
import yaml

def load_sweep_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_model_pth(model, run_id, method=None, sam_version=None, da_method=None, lr_method=None):
    base = f"{model}_run{run_id}" if sam_version is None \
           else f"sam{sam_version}_{method}_{model}_run{run_id}"
    if da_method is not None:
        base = f"{base}_{da_method}"
    if lr_method is not None:
        base = f"{base}_{lr_method}"
    return f"{base}.pth"

def build_command(model, run_id, sam_version, method, da_method, lr_method):
    cmd = []
    cmd.append("python")
    cmd.append(model['test_script'])

    cmd.append("--model_pth")
    model_pth = build_model_pth(model['name'], run_id, method, sam_version, da_method, lr_method)
    cmd.append(model_pth)

    if (sam_version is not None) and (method is not None):
        cmd.append("--sam_version")
        cmd.append(str(sam_version))
        cmd.append("--method")
        cmd.append(str(method))

    return cmd

def run_inference(cmd, cwd):
    print(f"Eseguendo comando: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERRORE: Il comando è terminato con un errore (codice {e.returncode}): {' '.join(cmd)}")
        exit(1)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="../configs/sweep.yaml", help="path to sweep config file")
    parser.add_argument("--model", type=str, default=None, help="specific model to run (overrides config file)")
    parser.add_argument("--run_id", type=int, default=None, help="specific run ID to execute (overrides config file)")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERRORE: FILE CONFIGURAZIONE NON TROVATO: {args.config}")
        exit(1)

    sweep = load_sweep_config(args.config)

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
        run_ids = [args.run_id]
    else:
        run_ids = list(range(1, sweep['testing']['runs'] + 1))
    
    for model in models_to_run:
        cwd = "../src/models/" + model['folder']
        print(f"Controllando esistenza cartella modello: {cwd}")
        
        if not os.path.exists(cwd):
            print(f"ERRORE: CARTELLA MODELLO NON TROVATA: {cwd}")
            exit(1)
        
        da_methods = sweep['testing'].get('da_methods', [None])   # asse DA (solo vanilla); senza chiave -> 1 run legacy
        lr_methods = sweep['testing'].get('lr_methods', [None])   # retrocompat: senza chiave -> 1 run legacy
        for run_id in run_ids:
            if model['has_aux']:
                for sam_version in sweep['testing']['sam_versions']:
                    for method in sweep['testing']['aug_methods']:
                        for lr_method in lr_methods:
                            cmd = build_command(model, run_id, sam_version, method, None, lr_method)
                            run_inference(cmd, cwd)
            else:
                for da_method in da_methods:
                    for lr_method in lr_methods:
                        cmd = build_command(model, run_id, None, None, da_method, lr_method)
                        run_inference(cmd, cwd)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
