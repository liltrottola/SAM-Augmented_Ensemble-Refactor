"""Single source of truth for run identity in the sweep.

Both run_training.py and run_inference.py import from here so that the
checkpoint filename produced at training time and the one looked up at
test time can never drift apart.
"""
from src.augmentation import OFFLINE_DA_METHODS


def build_model_name(model, run_id, method=None, sam_version=None,
                     da_method=None, lr_method=None):
    """Deterministic checkpoint stem (no extension) for one sweep point."""
    base = f"{model}" if sam_version is None \
           else f"sam{sam_version}_{method}_{model}"
    if da_method is not None:
        base = f"{base}_{da_method}"
    if lr_method is not None:
        base = f"{base}_{lr_method}"

    base = f"{base}_run{run_id}"
    return base


def is_valid_combo(model, da_method):
    """Whether a (model, da_method) pair is a real, trainable combination.

    Aux models only support the online DA axis; an offline DA (da1/da2)
    for an aux model is the not-yet-implemented "SAM regenerated over
    offline-augmented data" case and must be skipped in both runners.
    """
    if model["has_aux"] and da_method in OFFLINE_DA_METHODS:
        return False
    return True
