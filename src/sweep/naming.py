"""Single source of truth for run identity in the sweep.

Both run_training.py and run_inference.py import from here so that the
checkpoint filename produced at training time and the one looked up at
test time can never drift apart.
"""


def build_model_name(model, run_id, method=None, aux_source=None,
                     da_method=None, lr_method=None, priors=None, train_dataset=None):
    """Deterministic checkpoint stem (no extension) for one sweep point.

    aux_source is the folder name under aux_root_base (e.g. "sam1", "sam2",
    "da3mono-large")
    priors (N3 models) is the ordered list of "<folder>/<prior>" of registry.get_n3_priors: only the prior names go
    into the name, joined by "-" (e.g. segPrior-boundaryPrior-frst_prior-localnorm_hsnet_n3_da1_lrbase_run1)
    train_dataset (folder under datasets/) always goes first, e.g. TrainDataset_hsnet_da3_lrbase_run1
    
    """
    base = f"{model}" if aux_source is None \
           else f"{aux_source}_{method}_{model}"
    if priors is not None:
        prior_names = []
        for channel in priors:
            prior_names.append(channel.split("/")[-1])   # "<folder>/<prior>" -> "<prior>"
        base = f"{'-'.join(prior_names)}_{base}"
    if da_method is not None:
        base = f"{base}_{da_method}"
    if lr_method is not None:
        base = f"{base}_{lr_method}"

    base = f"{base}_run{run_id}"
    if train_dataset is not None:
        base = f"{train_dataset.strip('/').replace('/', '-')}_{base}"
    return base
