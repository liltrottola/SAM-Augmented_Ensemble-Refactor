import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import os

def calculate_dice(pred_tensor, gt_tensor):
    """Calculates the Dice score between prediction and ground truth, binarizing at 0.5."""
    pred = (pred_tensor >= 0.5).float()
    gt = (gt_tensor >= 0.5).float()
    
    pred_flat = pred.view(-1).numpy()
    gt_flat = gt.view(-1).numpy()
    
    intersection = (pred_flat * gt_flat).sum()
    smooth = 1e-8
    dice = (2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
    return float(dice)

def run_oracle(models_path, labels_root, dataset_name):
    # Identify all available models
    models = sorted([m for m in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, m))])
    image_files = sorted(os.listdir(os.path.join(models_path, models[0], dataset_name)))
    
    transform = transforms.ToTensor()
    dataset_w_dices = []
    
    sum_weights = np.zeros(len(models), dtype=float)
    max_weights = np.zeros(len(models), dtype=float)
    min_weights = np.ones(len(models), dtype=float) * float('inf')
    zero_counts = [0] * len(models)
    zero_files = [[] for _ in range(len(models))]
    image_scores = {}
    image_zero_counts = {}
    
    oraclew_out_dir = os.path.join("..", "output", "oracleW", dataset_name)
    os.makedirs(oraclew_out_dir, exist_ok=True)
    
    for img_name in image_files:
        # Load the Ground Truth (GT) mask
        gt_path = os.path.join(labels_root, "masks", img_name)
        gt_img = transform(Image.open(gt_path).convert("L"))
        
        image_dices = []
        image_preds = []
        
        # 1. Calculate Dice scores for each individual expert
        for m in models:
            pred_path = os.path.join(models_path, m, dataset_name, img_name)
            pred_img = transform(Image.open(pred_path).convert("L"))
            
            d = calculate_dice(pred_img, gt_img)
            image_dices.append(d)
            image_preds.append(pred_img)
       
        # 2. Sum of all Dice scores for weight normalization
        total_dice = sum(image_dices)
        
        # 3. Generate the Weighted Mask (OracleW)
        weighted_mask = torch.zeros_like(gt_img)
        for i in range(len(models)):
            # weighting formula: Score_i = Dice_i / SumOfDices
            weight = image_dices[i] / (total_dice + 1e-8)

            if weight < 0.0004:
                zero_counts[i] += 1
                zero_files[i].append(img_name)
                if dataset_name in ["CVC-ColonDB", "ETIS-LaribPolypDB"]:
                    image_zero_counts[img_name] = image_zero_counts.get(img_name, 0) + 1

            # updating counter vectors
            sum_weights[i] += weight
            max_weights[i] = max(max_weights[i], weight)
            min_weights[i] = min(min_weights[i], weight)
            
            # Accumulate the weighted contribution of the current mask
            weighted_mask += image_preds[i] * weight
            
        # 4. Calculate the final Dice score of the aggregated mask against the GT
        final_dice = calculate_dice(weighted_mask, gt_img)
        dataset_w_dices.append(final_dice)
        image_scores[img_name] = final_dice

        w_mask_img = (weighted_mask.squeeze().numpy() * 255).astype(np.uint8)
        Image.fromarray(w_mask_img).save(os.path.join(oraclew_out_dir, img_name))

    # 5. Compute the overall mean Dice across the dataset
    mDice = sum(dataset_w_dices) / len(dataset_w_dices)
    
    print(f"{dataset_name} mDICE(WWW): {mDice:.3f}")
    
    mean_weights = sum_weights / len(image_files)

    mean_weights = np.round(mean_weights, 3)
    max_weights = np.round(max_weights, 3)
    min_weights = np.round(min_weights, 3)

    print(f"Mean Weights: {mean_weights.tolist()}")
    print(f"Max Weights: {max_weights.tolist()}")
    print(f"Min Weights: {min_weights.tolist()}")
    print(f"Zero Weight Counts: {zero_counts}")
    print(f"Zero Weight Files: {zero_files}")
    if dataset_name in ["CVC-ColonDB", "ETIS-LaribPolypDB"]:
        print(f"\n--- Analysis of Experts at 0 per Image ({dataset_name}) ---")
        sorted_zeros = dict(sorted(image_zero_counts.items(), key=lambda item: item[1], reverse=True))
        for img, count in sorted_zeros.items():
            print(f"  {img}: {count}")
        print("-" * 50)

    return mDice, image_scores
