import argparse
import os
import subprocess
import yaml

def load_sweep_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_model_name(model, run_id, method=None, sam_version=None):
    if sam_version is None:
        return f"{model}_run{run_id}"
    
    return f"sam{sam_version}_{method}_{model}_run{run_id}"

def build_command(model, seed, run_id, sam_version, method, debug):
    cmd = []
    cmd.append("python")
    cmd.append(model['train_script'])
    
    cmd.append("--model_name")
    model_name = build_model_name(model['name'], run_id, method, sam_version)
    cmd.append(model_name)
    
    cmd.append("--seed")
    cmd.append(str(seed))

    if (sam_version is not None) and (method is not None):
        cmd.append("--sam_version")
        cmd.append(str(sam_version))
        cmd.append("--method")
        cmd.append(str(method))
    
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

    parser.add_argument("--config", type=str, default="../configs/sweep.yaml", help="path to sweep config file")
    parser.add_argument("--model", type=str, default=None, help="specific model to run (overrides config file)")
    parser.add_argument("--run_id", type=int, default=None, help="specific run ID to execute (overrides config file)")
    parser.add_argument("--debug", action='store_true', default=False, help="run in debug mode")

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
        run_ids = list(range(1, sweep['training']['runs'] + 1))

    for model in models_to_run:
        cwd = "../src/models/" + model['folder']
        print(f"Controllando esistenza cartella modello: {cwd}")
        
        if not os.path.exists(cwd):
            print(f"ERRORE: CARTELLA MODELLO NON TROVATA: {cwd}")
            exit(1)
        
        for run_id in run_ids:
            if model['has_aux']:
                for sam_version in sweep['training']['sam_versions']:
                    for method in sweep['training']['aug_methods']:
                        cmd = build_command(model, sweep['training']['seeds'][run_id - 1], run_id, sam_version, method, args.debug)
                        run_training(cmd, cwd)
            else:
                cmd = build_command(model, sweep['training']['seeds'][run_id - 1], run_id, None, None, args.debug)           
                run_training(cmd, cwd)
        
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()