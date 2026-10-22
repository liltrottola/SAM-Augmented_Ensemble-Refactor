import os
import sys
import time
import argparse
import yaml
import numpy as np
from PIL import Image

# import modules from src even if we are in scripts folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation import normal_prior_methods as npm


def check_dataset(file_names, out_dirs):
    """Every image must have exactly one map with its name in every prior folder (count and names)."""
    expected = {os.path.splitext(f)[0] for f in file_names}
    for prior_name, out_dir in out_dirs.items():
        saved = {os.path.splitext(f)[0] for f in os.listdir(out_dir) if f.endswith(".npy")}
        if saved != expected:
            raise RuntimeError(f"{out_dir}: {len(saved)} maps for {len(expected)} images, "
                               f"missing {sorted(expected - saved)[:5]}, extra {sorted(saved - expected)[:5]}")
        sample = np.load(os.path.join(out_dir, sorted(saved)[0] + ".npy"))
        print(f"   check {prior_name}: {len(saved)} maps = {len(expected)} images, names OK | "
              f"sample {sample.dtype} {sample.shape} min {sample.min():.3f} max {sample.max():.3f}")


def run_processing(config, limit=None, vis=False):
    dataset_root = config['paths']['dataset_root']
    normal_root = config['paths']['normal_root']
    saving_root = os.path.join(config['paths']['output_root'], config['paths']['prior_source'])
    priors = config['priors']
    for prior_name in priors:
        if not hasattr(npm, prior_name):
            raise ValueError(f"Prior '{prior_name}' not defined in normal_prior_methods.py")

    for dataset in config['datasets']['folders']:
        images_path = os.path.join(dataset_root, dataset, "images")
        normal_path = os.path.join(normal_root, dataset, "npy")
        print(f"------ {dataset}: images {images_path}, normals {normal_path}")

        file_names = sorted(f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')))
        if limit is not None:
            file_names = file_names[:limit]

        # every image must have its normal map
        missing = [f for f in file_names if not os.path.exists(os.path.join(normal_path, os.path.splitext(f)[0] + ".npy"))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} normal maps missing in {normal_path}, first: {missing[:5]}")

        out_dirs = {p: os.path.join(saving_root, dataset, p) for p in priors}
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)
        
        vis_dirs = {p: os.path.join(saving_root, dataset, "vis", p) for p in priors}
        if vis:
            for d in vis_dirs.values():
                os.makedirs(d, exist_ok=True)
    
        t_tot = 0.0
        for count, filename in enumerate(file_names, 1):
            name = os.path.splitext(filename)[0]
            N = np.load(os.path.join(normal_path, name + ".npy"))
            if N.ndim != 3 or N.shape[2] != 3:
                raise ValueError(f"{name}: invalid normal map shape {N.shape}, expected (H, W, 3)")

            t0 = time.time()
            for prior_name in priors:
                P = getattr(npm, prior_name)(N).astype(np.float32)
                np.save(os.path.join(out_dirs[prior_name], name + ".npy"), P)
                if vis:
                    png = (np.clip(P, 0, 1) * 255).round().astype(np.uint8)
                    Image.fromarray(png).save(os.path.join(vis_dirs[prior_name], name + ".png"))
            t1 = time.time()
            t_tot += t1 - t0

            print(f"(-) {dataset}: {count}/{len(file_names)}  {name}  {t1 - t0:.2f}s")

        print(f"====== {dataset}: {len(file_names)} images, mean {t_tot / max(len(file_names), 1):.2f} s/img")
        check_dataset(file_names, out_dirs)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(current_dir, '../configs/normal_prior_maps.yaml'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path)
    parser.add_argument('--limit', type=int, default=None, help='Only the first N images of each dataset (quick test)')
    parser.add_argument('--vis', action='store_true', help='Also save a grayscale PNG of every prior (debug)')
    opt = parser.parse_args()

    if not os.path.exists(opt.config):
        raise FileNotFoundError(f"Config file not found: {opt.config}")

    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    run_processing(config, opt.limit, opt.vis)

