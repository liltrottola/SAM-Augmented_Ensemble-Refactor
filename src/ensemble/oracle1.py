import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import os

def calculate_dice(pred_tensor, gt_tensor):
    pred = (pred_tensor >= 0.5).float()
    gt = (gt_tensor >= 0.5).float()
    
    pred_flat = pred.view(-1).numpy()
    gt_flat = gt.view(-1).numpy()
    
    intersection = (pred_flat * gt_flat).sum()
    smooth = 1e-8
    dice = (2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
    return dice

def run_oracle(models_path, labels_root, dataset_name):
    models = [m for m in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, m))]
    image_files = sorted(os.listdir(os.path.join(models_path, models[0], dataset_name)))
    
    transform = transforms.ToTensor()
    dataset_best_dices = []

    for img_name in image_files:
        gt_path = os.path.join(labels_root, "masks", img_name)
        gt_img = transform(Image.open(gt_path).convert("L"))
        
        image_scores = []
        for m in models:
            pred_path = os.path.join(models_path, m, dataset_name, img_name)
            pred_img = transform(Image.open(pred_path).convert("L"))
            
            score = calculate_dice(pred_img, gt_img)
            image_scores.append(score)
        
        dataset_best_dices.append(max(image_scores))

    mDice = sum(dataset_best_dices) / len(dataset_best_dices)
    print(f"{dataset_name} mDICE: {mDice:.3f}")
    return mDice
