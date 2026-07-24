import torch

# Canonical paper definitions. gamma = 0.1 for all (×0.1 step drop).
LR_METHODS = {
    "lra": {"init_lr": 1e-4, "milestones": [],   "gamma": 0.1},  # constant
    "lrb": {"init_lr": 5e-4, "milestones": [10],  "gamma": 0.1},  # ->5e-5 @10
    "lrc": {"init_lr": 5e-5, "milestones": [30],  "gamma": 0.1},  # ->5e-6 @30
}

def get_lr_method(name):
    """Return the {init_lr, milestones, gamma} dict for an LR method name."""
    if name is None or name.lower() == "lrbase":
        return None
    key = name.lower()

    if key not in LR_METHODS:
        raise ValueError(f"Unknown lr_method '{name}'. Expected one of {list(LR_METHODS)}, 'lrbase', or null.")
    
    return LR_METHODS[key]

def build_scheduler(optimizer, cfg):
    """Build a MultiStepLR from an LR-method cfg dict."""
    
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"]
    )