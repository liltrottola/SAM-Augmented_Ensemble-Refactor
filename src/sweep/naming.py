"""Single source of truth for run identity in the sweep.

Both run_training.py and run_inference.py import from here so that the
checkpoint filename produced at training time and the one looked up at
test time can never drift apart.
"""


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
