"""Single source of truth for run identity in the sweep.

Both run_training.py and run_inference.py import from here so that the
checkpoint filename produced at training time and the one looked up at
test time can never drift apart.
"""
from src.augmentation import OFFLINE_DA_METHODS


def build_model_name(model, run_id, method=None, aux_source=None,
                     da_method=None, lr_method=None):
    """Deterministic checkpoint stem (no extension) for one sweep point.

    aux_source is the folder name under aux_root_base (e.g. "sam1", "sam2",
    "da3mono-large")
    """
    base = f"{model}" if aux_source is None \
           else f"{aux_source}_{method}_{model}"
    if da_method is not None:
        base = f"{base}_{da_method}"
    if lr_method is not None:
        base = f"{base}_{lr_method}"

    base = f"{base}_run{run_id}"
    return base


def get_aux_sources(section):
    """(aux_source, method) pairs to sweep for aux models, from a training/testing section.

    aux_sources` maps each source folder to its own method list,  SAM methods and depth methods
    only exist under their own source.
    """
    sources = section["aux_sources"]
    return [(source, method) for source, methods in sources.items() for method in methods]


def is_valid_combo(model, da_method):
    """Whether a (model, da_method) pair is a real, trainable combination.

    Aux models only support the online DA axis; an offline DA (da1/da2)
    for an aux model is the not-yet-implemented "SAM regenerated over
    offline-augmented data" case and must be skipped in both runners.
    """
    if model["has_aux"] and da_method in OFFLINE_DA_METHODS:
        return False
    return True
