import os
import sys
import time
import argparse
import yaml
import numpy as np
import cv2
import skimage

# import modules from src even if we are in scripts folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation import normal_prior_methods as npm

# The methods are chosen by name from the yaml: each name is a function
# of normal_prior_methods.py with signature f(tI, priors) -> final uint8 image
# The priors are the .npy maps saved by run_normal_prior_maps.py.

def read_rgb(path):
    """grayscale -> 3 channels, RGBA -> RGB, uint8."""
    tI = skimage.io.imread(path)
    if tI.ndim == 2:
        tI = np.stack([tI] * 3, axis=-1)
    if tI.shape[2] == 4:
        tI = tI[:, :, :3]
    if tI.dtype != np.uint8:
        tI = skimage.img_as_ubyte(tI)
    return tI


def run_processing(config, limit=None):
    dataset_root = config['paths']['dataset_root']
    prior_root = os.path.join(config['paths']['prior_root'], config['paths']['prior_source'])
    saving_root = os.path.join(config['paths']['output_root'], config['paths']['aux_source'])
    prior_names = config['priors']
    methods = config['augmentation']['methods'] or []

    if not methods:
        raise ValueError("No methods in augmentation.methods of the config")
    for m in methods:
        if not hasattr(npm, m):  # check if the method exists in normal_prior_methods.py
            raise ValueError(f"Method '{m}' not defined in normal_prior_methods.py")

    for dataset in config['datasets']['folders']:
        images_path = os.path.join(dataset_root, dataset, "images")
        prior_paths = {p: os.path.join(prior_root, dataset, p) for p in prior_names}
        print(f"------ {dataset}: images {images_path}, priors {os.path.join(prior_root, dataset)} {prior_names}")

        file_names = sorted(f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')))
        if limit is not None:
            file_names = file_names[:limit]

        # the loader pairs image and aux by name: every image must have all its priors
        for p, path in prior_paths.items():
            missing = [f for f in file_names if not os.path.exists(os.path.join(path, os.path.splitext(f)[0] + ".npy"))]
            if missing:
                raise FileNotFoundError(f"{len(missing)} '{p}' maps missing in {path} "
                                        f"(run run_normal_prior_maps.py first), first: {missing[:5]}")

        out_dirs = {m: os.path.join(saving_root, m, dataset, "images") for m in methods}
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)

        t_prior, t_methods = 0.0, 0.0
        for count, filename in enumerate(file_names, 1):
            name = os.path.splitext(filename)[0]
            tI = read_rgb(os.path.join(images_path, filename))
            h, w = tI.shape[:2]

            # priors already saved at the image resolution by run_normal_prior_maps.py:
            t0 = time.time()
            priors = {}
            for p, path in prior_paths.items():
                prior = np.load(os.path.join(path, name + ".npy")).astype(np.float32)
                if prior.shape != (h, w):
                    prior = cv2.resize(prior, (w, h), interpolation=cv2.INTER_LINEAR)
                priors[p] = prior
            t1 = time.time()

            for m in methods:
                aug_img = getattr(npm, m)(tI, priors)
                # always PNG (lossless) with the image name: the aux loaders match by name
                skimage.io.imsave(os.path.join(out_dirs[m], name + ".png"), aug_img, check_contrast=False)
            t2 = time.time()
            t_prior += t1 - t0
            t_methods += t2 - t1

            print(f"(-) {dataset}: {count}/{len(file_names)}  {name}  priors {t1 - t0:.2f}s  methods+save {t2 - t1:.2f}s")

        n = max(len(file_names), 1)
        print(f"====== {dataset}: {len(file_names)} images, mean priors {t_prior / n:.2f} s/img, "
              f"mean methods+save {t_methods / n:.2f} s/img")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(current_dir, '../configs/normal_augmentation.yaml'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path)
    parser.add_argument('--limit', type=int, default=None, help='Only the first N images of each dataset (quick test)')
    opt = parser.parse_args()

    if not os.path.exists(opt.config):
        raise FileNotFoundError(f"Config file not found: {opt.config}")

    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    run_processing(config, opt.limit)
