import torch
import sys
import os

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print("WARNING: SAM2 library not found.")

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("WARNING: SAM1 library not found.")
    
def load_sam_model(version, checkpoint_path, sam_config, device='cuda'):
    """
    Load the SAM model based on the specified version and checkpoint path.

    Args:
        version (int): The version of the SAM model to load (1 or 2).
        checkpoint_path (str): The path to the model checkpoint.
        device (str): The device to load the model onto ('cuda' or 'cpu').
    """
    if not os.path.exists(checkpoint_path):
             raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
    
    if version == 2:
        #model_cfg = "sam2_hiera_l.yaml" 
        model_cfg = sam_config['model_cfg']

        print(f"Loading SAM2 from {checkpoint_path}...")
        sam_model = build_sam2(model_cfg, checkpoint_path, device=device, apply_postprocessing=False)
        return SAM2AutomaticMaskGenerator(sam_model)

         #to load a finetuned model
            #ft_model=torch.load("/.../model.torch")
            #mask_generator.model.load_state_dict(ft_model))
    elif version == 1:
        sam_model = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam_model.to(device=device)
        return SamAutomaticMaskGenerator(
            sam_model, 
            crop_nms_thresh=sam_config['thresholds']['crop_nms'], 
            box_nms_thresh=sam_config['thresholds']['box_nms'], 
            pred_iou_thresh=sam_config['thresholds']['pred_iou']
        )
    else:
        raise ValueError("(x) Error: insert a valid SAM version (1,2)")