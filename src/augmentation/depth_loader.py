import torch
import sys
import os

try:
    from depth_anything_3.api import DepthAnything3
    DA3_AVAILABLE = True
except ImportError:
    print("WARNING: Depth Anything 3 library not found.")
    DA3_AVAILABLE = False

def load_depth_model(model_dir, device='cuda'):
    if not DA3_AVAILABLE:
        raise RuntimeError("Depth Anything 3 not available. Please install or active the environment venv_dav3.")
    
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Checkpoint not found: {model_dir}")
    print(f"Loading Depth Anything 3 model from {model_dir}...")

    depth_model = DepthAnything3.from_pretrained(model_dir)
    depth_model.to(device=device)

    print("Depth Anything 3 model loaded successfully.")
    return depth_model