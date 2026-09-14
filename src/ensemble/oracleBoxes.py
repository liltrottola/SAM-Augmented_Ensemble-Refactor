import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw
from scipy.ndimage import label, find_objects

def calculate_dice(pred_tensor, gt_tensor):
    """Calculates the Dice score binarizing at a 0.5 threshold."""
    pred = (pred_tensor >= 0.5).float()
    gt = (gt_tensor >= 0.5).float()

    intersection = (pred * gt).sum()
    smooth = 1e-8
    dice = (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    return float(dice)

def get_bounding_boxes(gt_tensor):
    """Finds multiple unconnected polyps using Connected Component Analysis."""
    gt_np = (gt_tensor.squeeze().numpy() > 0.5).astype(int)
    labeled_array, num_features = label(gt_np)
    
    boxes = []
    if num_features > 0:
        slices = find_objects(labeled_array)
        for s in slices:
            if s is not None:
                y_slice, x_slice = s
                boxes.append({
                    'y_min': y_slice.start,
                    'y_max': y_slice.stop - 1,
                    'x_min': x_slice.start,
                    'x_max': x_slice.stop - 1
                })
    return boxes

def apply_bounding_boxes(image_tensor, boxes):
    """Zeros out all pixels outside the provided bounding boxes."""
    if not boxes:
        return torch.zeros_like(image_tensor)
        
    box_mask = torch.zeros_like(image_tensor)
    for b in boxes:
        box_mask[:, b['y_min']:b['y_max']+1, b['x_min']:b['x_max']+1] = 1.0
        
    return image_tensor * box_mask

def save_visualization(raw_mean_tensor, boxes, save_path):
    """Saves the raw ensemble prediction with GT bounding boxes drawn in red."""
    mean_np = (raw_mean_tensor.squeeze().numpy() * 255).astype(np.uint8)
    img_rgb = Image.fromarray(mean_np).convert("RGB")
    
    draw = ImageDraw.Draw(img_rgb)
    for b in boxes:
        shape = [b['x_min'], b['y_min'], b['x_max'], b['y_max']]
        draw.rectangle(shape, outline="red", width=3)
        
    img_rgb.save(save_path)

def run_oracle(models_path, labels_root, dataset_name):
    """Main execution function for OracleBoxes."""
    models = sorted([m for m in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, m))])
    image_files = sorted(os.listdir(os.path.join(models_path, models[0], dataset_name)))
    
    transform = transforms.ToTensor()
    dataset_dices = []
    image_scores = {}

    # Setup visualization folders
    vis_folder_mean = os.path.join("..", "output", "oracleBoxes", dataset_name, "mean")
    vis_folder_gt = os.path.join("..", "output", "oracleBoxes", dataset_name, "gt")
    os.makedirs(vis_folder_mean, exist_ok=True)
    os.makedirs(vis_folder_gt, exist_ok=True)

    for img_name in image_files:
        gt_path = os.path.join(labels_root, "masks", img_name)
        gt_img = transform(Image.open(gt_path).convert("L"))
        
        # Phase 1: Find all bounding boxes (handles multiple polyps)
        boxes = get_bounding_boxes(gt_img)
        
        # Phase 2: Compute the raw ensemble mean across all models
        ensemble_sum = torch.zeros_like(gt_img)
        for m in models:
            pred_path = os.path.join(models_path, m, dataset_name, img_name)
            pred_img = transform(Image.open(pred_path).convert("L"))
            ensemble_sum += pred_img
            
        raw_ensemble_mean = ensemble_sum / len(models)
        
        # Phase 3: Save the visualizations (mean and GT with red boxes)
        save_path_mean = os.path.join(vis_folder_mean, img_name)
        save_visualization(raw_ensemble_mean, boxes, save_path_mean)
        
        save_path_gt = os.path.join(vis_folder_gt, img_name)
        save_visualization(gt_img, boxes, save_path_gt)

        # Phase 4: Apply the bounding box filter to remove external noise
        final_boxed_mean = apply_bounding_boxes(raw_ensemble_mean, boxes)
        
        # Phase 5: Evaluate the Dice score on the filtered mask
        final_dice = calculate_dice(final_boxed_mean, gt_img)
        
        dataset_dices.append(final_dice)
        image_scores[img_name] = final_dice

    mDice = sum(dataset_dices) / len(dataset_dices)
    return mDice, image_scores
