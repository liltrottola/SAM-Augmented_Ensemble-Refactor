import os
import sys
import time
import argparse
import yaml
import numpy as np

# import modules from src even if we are in scripts folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation import depth_prior_methods as dpm

# The priors are chosen by name from the yaml, as in run_sam_augmentation.py: each name is a function
# of depth_prior_methods.py with signature f(depth) -> (H, W) float map, and also the output folder name.
# Same functions (and same per-image normalization, no FOV mask) as the aux images of run_depth_augmentation.py.


def check_dataset(file_names, out_dirs):
    """Every image must have exactly one map with its name in every prior folder (count and names)."""
    expected = {os.path.splitext(f)[0] for f in file_names}
    for prior_name, out_dir in out_dirs.items():
        saved = {os.path.splitext(f)[0] for f in os.listdir(out_dir) if f.endswith(".npy")}
        if saved != expected:
            raise RuntimeError(f"{out_dir}: {len(saved)} maps for {len(expected)} images, "
                               f"missing {sorted(expected - saved)[:5]}, extra {sorted(saved - expected)[:5]}")

        # dtype / shape / range on the first map
        sample = np.load(os.path.join(out_dir, sorted(saved)[0] + ".npy"))
        print(f"   check {prior_name}: {len(saved)} maps = {len(expected)} images, names OK | "
              f"sample {sample.dtype} {sample.shape} min {sample.min():.3f} max {sample.max():.3f}")


def run_processing(config, limit=None):
    dataset_root = config['paths']['dataset_root']
    depth_root = config['paths']['depth_root']
    saving_root = os.path.join(config['paths']['output_root'], config['paths']['prior_source'])
    priors = config['priors']
    for prior_name in priors:
        if not hasattr(dpm, prior_name):  # check if the prior exists in depth_prior_methods.py
            raise ValueError(f"Prior '{prior_name}' not defined in depth_prior_methods.py")

    for dataset in config['datasets']['folders']:
        images_path = os.path.join(dataset_root, dataset, "images")
        depth_path = os.path.join(depth_root, dataset, "npy")
        print(f"------ {dataset}: images {images_path}, depth {depth_path}")

        file_names = sorted(f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')))
        if limit is not None:
            file_names = file_names[:limit]

        # every image must have its depth
        missing = [f for f in file_names if not os.path.exists(os.path.join(depth_path, os.path.splitext(f)[0] + ".npy"))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} depth maps missing in {depth_path}, first: {missing[:5]}")

        out_dirs = {p: os.path.join(saving_root, dataset, p) for p in priors}
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)

        t_tot = 0.0
        for count, filename in enumerate(file_names, 1):
            name = os.path.splitext(filename)[0]
            D = np.load(os.path.join(depth_path, name + ".npy"))

            # priors at DA3 resolution, NOT resized: the N->3 loader resizes them to the image (bilinear)
            t0 = time.time()
            for prior_name in priors:
                P = getattr(dpm, prior_name)(D).astype(np.float32)
                np.save(os.path.join(out_dirs[prior_name], name + ".npy"), P)
            t1 = time.time()
            t_tot += t1 - t0

            print(f"(-) {dataset}: {count}/{len(file_names)}  {name}  {t1 - t0:.2f}s")

        print(f"====== {dataset}: {len(file_names)} images, mean {t_tot / max(len(file_names), 1):.2f} s/img")
        check_dataset(file_names, out_dirs)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(current_dir, '../configs/depth_prior_maps.yaml'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path)
    parser.add_argument('--limit', type=int, default=None, help='Only the first N images of each dataset (quick test)')
    opt = parser.parse_args()

    if not os.path.exists(opt.config):
        raise FileNotFoundError(f"Config file not found: {opt.config}")

    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    run_processing(config, opt.limit)
