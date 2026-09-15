import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from PIL import Image, ImageDraw

def calculate_dice(pred_tensor, gt_tensor):
    """
    Calculates the Dice score between prediction and ground truth.
    Both tensors are binarized using a 0.5 threshold before the calculation.
    """
    # Binarize predictions and ground truth arrays (1 for polyp, 0 for background)
    pred = (pred_tensor >= 0.5).float()
    gt = (gt_tensor >= 0.5).float()
    
    # Calculate intersection between prediction and ground truth masks
    intersection = (pred * gt).sum()
    
    # Smoothing factor to prevent division by zero in empty masks
    smooth = 1e-8
    
    # Apply the standard Dice coefficient formula: 2 * |X ∩ Y| / (|X| + |Y|)
    dice = (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    
    return float(dice)

def apply_bounding_box(image_tensor, gt_tensor):
    """
    Extracts the bounding box coordinates from the Ground Truth (GT) and 
    zeros out all the pixels in the predicted tensor that fall outside this box.
    """
    # Find the coordinates (indices) of all pixels representing the polyp (values > 0.5)
    non_zero_indices = torch.nonzero(gt_tensor > 0.5)
    
    # Safety check: if there is no polyp in the GT (completely black mask), 
    # return a completely zeroed tensor to avoid out-of-bounds errors
    if non_zero_indices.numel() == 0:
        return torch.zeros_like(image_tensor)
        
    # Extract the bounding box limits: min/max for y (height) and x (width)
    y_min = torch.min(non_zero_indices[:, 1])
    y_max = torch.max(non_zero_indices[:, 1])
    x_min = torch.min(non_zero_indices[:, 2])
    x_max = torch.max(non_zero_indices[:, 2])
    
    # Create an empty mask (zeros) with the exact same shape as the Ground Truth
    box_mask = torch.zeros_like(gt_tensor)
    
    # Fill the extracted bounding box area with ones (1.0)
    box_mask[:, y_min:y_max+1, x_min:x_max+1] = 1.0
    
    # Multiply the prediction by the mask: pixels outside the bounding box 
    # become zero, effectively removing external noise
    return image_tensor * box_mask

def run_oracle(models_path, labels_root, dataset_name):
    """
    Main execution function called dynamically by run_oracle.py. 
    It iterates over images, applies the bounding box filter, computes 
    the average ensemble, and evaluates the mean Dice score.
    """
   
   #Identify all available expert models sorted alphabetically
    models = sorted([m for m in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, m))])
    
    # Get the list of all image filenames from the first model's output folder
    image_files = sorted(os.listdir(os.path.join(models_path, models[0], dataset_name)))
    
    transform = transforms.ToTensor()
    dataset_dices = []
    image_scores = {}

    
    oraclebox_out_dir = os.path.join("..", "output", "oracleBox", dataset_name)
    os.makedirs(oraclebox_out_dir, exist_ok=True)

    # Process each image in the dataset individually
    for img_name in image_files:
        
        # Load the Ground Truth image and convert it to a PyTorch tensor (Grayscale)
        gt_path = os.path.join(labels_root, "masks", img_name)
        gt_img = transform(Image.open(gt_path).convert("L"))
        
        # Initialize an empty tensor to accumulate the sum of all model predictions
        ensemble_sum = torch.zeros_like(gt_img)
        
        # Iterate through each model to process its specific prediction
        for m in models:
            pred_path = os.path.join(models_path, m, dataset_name, img_name)
            pred_img = transform(Image.open(pred_path).convert("L"))
            
            # Apply the OracleBox filter: remove noise outside the GT bounding box
            boxed_pred = apply_bounding_box(pred_img, gt_img)
            
            # Add the filtered prediction to the ensemble accumulator
            ensemble_sum += boxed_pred
            
        # Compute the final ensemble mask by averaging the accumulated predictions
        ensemble_mean = ensemble_sum / len(models)
        
        # Evaluate the Dice score of the final ensemble mask against the Ground Truth
        final_dice = calculate_dice(ensemble_mean, gt_img)
        dataset_dices.append(final_dice)
        image_scores[img_name] = final_dice

        mean_np = (ensemble_mean.squeeze().numpy() * 255).astype(np.uint8)
        img_rgb = Image.fromarray(mean_np).convert("RGB")
        non_zero_indices = torch.nonzero(gt_img > 0.5)
        if non_zero_indices.numel() > 0:
            y_min = torch.min(non_zero_indices[:, 1]).item()
            y_max = torch.max(non_zero_indices[:, 1]).item()
            x_min = torch.min(non_zero_indices[:, 2]).item()
            x_max = torch.max(non_zero_indices[:, 2]).item()
            draw = ImageDraw.Draw(img_rgb)
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        img_rgb.save(os.path.join(oraclebox_out_dir, img_name))

    # Compute the mean Dice score across all images in the current dataset
    mDice = sum(dataset_dices) / len(dataset_dices)
    
    return mDice, image_scores
