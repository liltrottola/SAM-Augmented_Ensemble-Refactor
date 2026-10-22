import os
import sys
import torch


def load_dsine_model(dsine_root, config_txt, device='cuda'):
    dsine_root = os.path.abspath(dsine_root)
    config_txt = os.path.abspath(config_txt)

    if not os.path.isdir(dsine_root):
        raise FileNotFoundError(f"DSINE repo not found: {dsine_root} (run setup_DSINE.sh)")
    if not os.path.isfile(config_txt):
        raise FileNotFoundError(f"DSINE config not found: {config_txt}")

    # DSINE is not an installable package: its modules live in its own root
    if dsine_root not in sys.path:
        sys.path.insert(0, dsine_root)
    try:
        import utils.utils as utils
        import projects.dsine.config as config
    except ImportError as e:
        raise RuntimeError("DSINE not importable. Please activate the environment venv_dsine.") from e

    print(f"Loading DSINE model from {config_txt}...")
    cwd, old_argv = os.getcwd(), sys.argv
    # config.get_args reads sys.argv, and paths in the .txt are relative to projects/dsine/
    os.chdir(os.path.join(dsine_root, "projects", "dsine"))
    sys.argv = [old_argv[0], config_txt]
    try:
        args = config.get_args(test=True)
        if not os.path.exists(args.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

        if args.NNET_architecture == 'v00':
            from models.dsine.v00 import DSINE_v00 as DSINE
        elif args.NNET_architecture == 'v01':
            from models.dsine.v01 import DSINE_v01 as DSINE
        elif args.NNET_architecture == 'v02':
            from models.dsine.v02 import DSINE_v02 as DSINE
        elif args.NNET_architecture == 'v02_kappa':
            from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
        else:
            raise ValueError(f"invalid arch: {args.NNET_architecture}")

        model = DSINE(args).to(device)
        model = utils.load_checkpoint(args.ckpt_path, model)
    finally:
        os.chdir(cwd)
        sys.argv = old_argv

    print("DSINE model loaded successfully.")
    return model.eval()
