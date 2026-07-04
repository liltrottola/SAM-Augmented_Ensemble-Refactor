import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import os

def calculate_dice(pred_tensor, gt_tensor):
    """Calcola il Dice score tra predizione e ground truth binarizzando a 0.5."""
    pred = (pred_tensor >= 0.5).float()
    gt = (gt_tensor >= 0.5).float()
    
    pred_flat = pred.view(-1).numpy()
    gt_flat = gt.view(-1).numpy()
    
    intersection = (pred_flat * gt_flat).sum()
    smooth = 1e-8
    dice = (2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
    return float(dice)

def run_oracle(models_path, labels_root, dataset_name):
    # Identifica tutti i modelli disponibili
    models = sorted([m for m in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, m))])
    image_files = sorted(os.listdir(os.path.join(models_path, models[0], dataset_name)))
    
    transform = transforms.ToTensor()
    dataset_w_dices = []

    for img_name in image_files:
        # Carica la Ground Truth (GT)
        gt_path = os.path.join(labels_root, "masks", img_name)
        gt_img = transform(Image.open(gt_path).convert("L"))
        
        image_dices = []
        image_preds = []
        
        # 1. Calcolo dei Dice score per ogni singolo esperto
        for m in models:
            pred_path = os.path.join(models_path, m, dataset_name, img_name)
            pred_img = transform(Image.open(pred_path).convert("L"))
            
            d = calculate_dice(pred_img, gt_img)
            image_dices.append(d)
            image_preds.append(pred_img) # Salviamo la maschera per pesarla
        
        # 2. Somma dei Dice per la normalizzazione dei pesi
        total_dice = sum(image_dices)
        
        # 3. Creazione della Maschera Pesata (OracleW)
        weighted_mask = torch.zeros_like(gt_img)
        for i in range(len(models)):
            # Applichiamo la formula: Score_i = Dice_i / SommaDeiDice
            weight = image_dices[i] / (total_dice + 1e-8)
            
            # Sommiamo il contributo della maschera pesata
            weighted_mask += image_preds[i] * weight
            
        # 4. Calcolo del Dice finale dell'immagine aggregata rispetto alla GT
        final_dice = calculate_dice(weighted_mask, gt_img)
        dataset_w_dices.append(final_dice)

    # 5. Media complessiva sul dataset
    mDice = sum(dataset_w_dices) / len(dataset_w_dices)
    
    # Stampa e return compatibili al 100% con il runner dinamico
    print(f"{dataset_name} mDICE(WWW): {mDice:.3f}")
    return mDice
