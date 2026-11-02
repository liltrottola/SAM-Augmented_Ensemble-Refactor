import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..' , '..' , '..')) # to import from src

from src.augmentation import da3

'''
DataLoader for the N->3 block: input = RGB + K one-channel priors (N = 3 + K), stacked in one tensor.

- channels 0-2 are always the RGB, then the K priors in the order of prior_dirs (= order of `channels` in the yaml);
- every prior is a .npy float map at its model resolution (SAM2, DA3, ...), one folder per prior,
  file name = image name (<name>.npy), paired with the image BY NAME, not by position;
- each prior is resized (bilinear) to the image resolution, then follows the same Resize of the RGB;
- normalization: ImageNet only on the RGB; priors as they are if prior_stats is None, otherwise standardized
  with prior_stats = {"mean": [...], "std": [...]} (one value per prior, to be computed ONLY on the TrainDataset;
  not used for now). As they are = depth priors in [0,1] (normalized per image), SAM segPrior raw = sum of the
  stability scores of the overlapping masks (~0.9 per mask, max per image ~1-5, not normalized);
- online augmentation: None/"noda" = RGB and GT with exactly the same transforms as the vanilla loader in dataloader.py;
  "da3" = RGB and GT with the same da3 transforms as the vanilla loader (geometric + photometric on the RGB),
  priors with ONLY the geometric part (same rotation and flips, same seed), photometric changes only on the RGB;
- K = 0 (no priors): same output as PolypDataset / test_dataset.
'''

IMG_EXT = ('.jpg', '.png')


def list_by_name(root, extensions):
    """{name without extension: full path} of the files in root with one of the extensions."""
    return {os.path.splitext(f)[0]: os.path.join(root, f) for f in os.listdir(root) if f.endswith(extensions)}


def pair_by_name(image_root, gt_root, prior_dirs):
    """
        Pairs every image with its GT and its priors by file name.
        Explicit error if a GT or a prior is missing (instead of silently shifting the pairs as sorted lists do).
        Returns, sorted by name: image paths, GT paths and, per image, the list of prior paths (same order as prior_dirs).
    """
    images = list_by_name(image_root, IMG_EXT)
    gts = list_by_name(gt_root, IMG_EXT + ('.tif',))
    priors = [list_by_name(d, ('.npy',)) for d in prior_dirs]

    names = sorted(images)
    missing_gt = [n for n in names if n not in gts]
    if missing_gt:
        raise FileNotFoundError(f"{len(missing_gt)} GT missing in {gt_root}, first: {missing_gt[:5]}")
    for d, p in zip(prior_dirs, priors):
        missing = [n for n in names if n not in p]
        if missing:
            raise FileNotFoundError(f"{len(missing)} priors missing in {d}, first: {missing[:5]}")

    image_paths = [images[n] for n in names]
    gt_paths = [gts[n] for n in names]
    prior_paths = [[p[n] for p in priors] for n in names]
    return image_paths, gt_paths, prior_paths


def load_priors(paths, size, resize, prior_stats=None):
    """
        The K priors of one image as a (K, S, S) float tensor:
        .npy at its model resolution -> bilinear to the image resolution size = (w, h) -> same Resize as the RGB
        -> standardized only if prior_stats is given.
    """
    w, h = size
    priors = []
    for path in paths:
        p = np.load(path).astype(np.float32)
        p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)  # same as run_depth_augmentation.py / aux images
        priors.append(resize(torch.from_numpy(p).unsqueeze(0)))
    priors = torch.cat(priors, dim=0)

    if prior_stats is not None:
        mean = torch.tensor(prior_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(prior_stats["std"], dtype=torch.float32).view(-1, 1, 1)
        priors = (priors - mean) / std
    return priors


class PolypDatasetN3(data.Dataset):
    """
    Train dataset: returns image (3 + K, trainsize, trainsize) = [RGB ImageNet-normalized, K priors]
    and gt (1, trainsize, trainsize), as PolypDataset returns (image, gt).
    """
    def __init__(self, image_root, gt_root, prior_dirs, trainsize, prior_stats=None, augmentation=None):
        self.trainsize = trainsize
        self.prior_stats = prior_stats
        self.augmentation = augmentation
        self.images, self.gts, self.priors = pair_by_name(image_root, gt_root, prior_dirs)
        self.filter_files()
        self.size = len(self.images)

        print(f"N->3 loader: {self.size} images, {len(prior_dirs)} priors -> N = {3 + len(prior_dirs)} channels, "
              f"priors {'standardized' if prior_stats is not None else 'as they are (no standardization)'}")
        print("online augmentation:", self.augmentation)

        # same transforms as PolypDataset with no online augmentation; always kept, also with da3,
        # for plain_item (the not augmented samples used to save the block output)
        self.plain_img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        self.plain_gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        # Resize of the priors, applied on a (1, H, W) float tensor
        self.prior_resize = transforms.Resize((self.trainsize, self.trainsize))

        if self.augmentation == 'da3':
            # same da3 as PolypDataset for RGB and GT; priors: only the geometric part
            self.img_transform, self.gt_transform = da3.get_da3_transforms(self.trainsize)
            self.prior_transform = da3.get_da3_geometric_transforms()
        else:
            print('no online augmentation')
            self.img_transform = self.plain_img_transform
            self.gt_transform = self.plain_gt_transform
            self.prior_transform = None

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        size = image.size  # (w, h) of the image, before the Resize

        # da3 as PolypDataset: same seed before every transform, so that the random geometric transforms
        # (rotation angle, flips) are the same for RGB, GT and priors; no reseeding without augmentation (as before)
        augment = self.augmentation == 'da3'
        if augment:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
        image = self.img_transform(image)

        if augment:
            random.seed(seed)
            torch.manual_seed(seed)
        gt = self.gt_transform(gt)

        # K = 0 (only RGB): same output as PolypDataset
        if self.priors[index]:
            priors = load_priors(self.priors[index], size, self.prior_resize, self.prior_stats)
            if augment:
                # all the K priors together: one tensor, so they all get the same rotation and flips
                random.seed(seed)
                torch.manual_seed(seed)
                priors = self.prior_transform(priors)
            image = torch.cat([image, priors], dim=0)

        return image, gt

    def plain_item(self, index):
        """(image, gt) of index WITHOUT online augmentation, also when the loader uses da3 (fixed examples for the figures)."""
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        size = image.size
        image = self.plain_img_transform(image)
        gt = self.plain_gt_transform(gt)
        if self.priors[index]:
            priors = load_priors(self.priors[index], size, self.prior_resize, self.prior_stats)
            image = torch.cat([image, priors], dim=0)
        return image, gt

    def filter_files(self):
        #keep only the images with the same size as their GT
        assert len(self.images) == len(self.gts) == len(self.priors)
        images = []
        gts = []
        priors = []
        for img_path, gt_path, prior_paths in zip(self.images, self.gts, self.priors):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                priors.append(prior_paths)
        if len(images) != len(self.images):
            print(f"filter_files: {len(self.images) - len(images)} images dropped (image size != GT size)")
        self.images = images
        self.gts = gts
        self.priors = priors

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader_n3(image_root, gt_root, prior_dirs, batchsize, trainsize, shuffle=True, num_workers=1, pin_memory=True, prior_stats=None, augmentation=None):

    dataset = PolypDatasetN3(image_root, gt_root, prior_dirs, trainsize, prior_stats, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset_n3:
    """
    Test dataset: as test_dataset, load_data returns image (1, 3 + K, testsize, testsize),
    the GT at its original size (PIL, not resized) and the name.
    prior_stats must be the same used in training (None if the model was trained without standardization).
    """
    def __init__(self, image_root, gt_root, prior_dirs, testsize, prior_stats=None):
        self.testsize = testsize
        self.prior_stats = prior_stats
        self.images, self.gts, self.priors = pair_by_name(image_root, gt_root, prior_dirs)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.prior_resize = transforms.Resize((self.testsize, self.testsize))
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        size = image.size
        image = self.transform(image)

        if self.priors[self.index]:
            priors = load_priors(self.priors[self.index], size, self.prior_resize, self.prior_stats)
            image = torch.cat([image, priors], dim=0)
        image = image.unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        name = os.path.basename(self.images[self.index])
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
