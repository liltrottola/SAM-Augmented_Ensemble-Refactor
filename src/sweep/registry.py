import sys
import yaml

from src.augmentation import OFFLINE_DA_METHODS, ONLINE_DA_METHODS, NO_DA
from src.training.lr_schedules import LR_METHODS


def fail(message, valid=None):
    """Print the error (and the valid values) and stop, as the other errors of the runners."""
    print(f"ERROR: {message}")
    if valid is not None:
        print(f"Valid: {', '.join(str(v) for v in valid)}")
    sys.exit(1)


def load_registry(registry_path):
    """Load the registry (configs/sweep_registry.yaml)."""
    with open(registry_path, 'r') as f:
        registry = yaml.safe_load(f)
    return registry


def get_models(registry, sweep_models, cli_model=None):
    """Registry entries (folder, scripts, config, has_aux) of the models to launch.
    --model (one) replaces the models of the sweep; the names are checked against the registry."""

    # names to launch: the ones of the sweep, or only the one given with --model
    if cli_model is None:
        names = sweep_models
    else:
        names = [cli_model]

    models = []
    for name in names:
        # look for the model with this name in the registry
        found = None
        for model in registry['models']:
            if model['name'] == name:
                found = model

        # not in the registry: stop, with the valid names
        if found is None:
            valid = []
            for model in registry['models']:
                valid.append(model['name'])
            fail(f"unknown model: {name}", valid)

        models.append(found)
    return models


def get_aux_pairs(registry, sweep_section, cli_source=None, cli_methods=None):
    """(aux_source, method) pairs for the aux models.
    - no flag:                     every source of the sweep, with its methods in the sweep
    - --aux_source:                that source, with its methods in the sweep
    - --aux_source --aux_methods:  that source, with those methods
    Every source and method is checked against the registry."""
    registry_sources = registry['aux_sources']
    sweep_sources = sweep_section.get('aux_sources') or {}

    # --aux_methods only makes sense inside one source
    if cli_methods is not None and cli_source is None:
        fail("--aux_methods needs --aux_source (e.g. --aux_source sam2_stab085 --aux_methods SAMAug RG_logits)")

    # sources to launch: the ones of the sweep, or only the one given with --aux_source
    if cli_source is None:
        sources = list(sweep_sources.keys())
    else:
        sources = [cli_source]

    pairs = []
    for source in sources:
        # the source must exist in the registry
        if source not in registry_sources:
            fail(f"unknown aux source: {source}", list(registry_sources.keys()))

        # methods: the ones given with --aux_methods, or the ones of this source in the sweep
        if cli_methods is not None:
            methods = cli_methods
        elif source in sweep_sources:
            methods = sweep_sources[source]
        else:
            fail(f"aux source '{source}' has no methods in the sweep config: pass them with --aux_methods")

        # every method must exist in this source of the registry
        for method in methods:
            if method not in registry_sources[source]:
                fail(f"unknown method '{method}' for aux source '{source}'", registry_sources[source])
            pairs.append((source, method))
    return pairs


def get_da_methods(sweep_section, cli_values=None):
    """DA methods to launch: the ones given with --da_method, or da_methods of the sweep.
    Checked against the DA of the code (src/augmentation): a typo would otherwise train with no augmentation.
    Neither of them -> [None]: 1 run with the DA of the model yaml (printed by the runner, not in the checkpoint name)."""
    # valid names: no augmentation, the online DA and the offline ones
    valid = [NO_DA] + sorted(ONLINE_DA_METHODS) + sorted(OFFLINE_DA_METHODS)

    if cli_values is not None:
        values = cli_values
    elif 'da_methods' in sweep_section:
        values = sweep_section['da_methods']
    else:
        print("WARNING: no da_methods in the sweep config and no --da_method: "
              "the DA of each model yaml is used (not in the checkpoint name)")
        values = [None]

    for value in values:
        if value is not None and value not in valid:
            fail(f"unknown DA method: {value}", valid)
    return values


def get_lr_methods(sweep_section, cli_values=None):
    """LR methods to launch: the ones given with --lr_method, or lr_methods of the sweep.
    Checked against the LR methods of src/training/lr_schedules.py + "lrbase" (the legacy one of the model yaml).
    Neither of them -> [None]: 1 run with the LR of the model yaml (printed by the runner, not in the checkpoint name)."""
    valid = ["lrbase"] + list(LR_METHODS.keys())

    if cli_values is not None:
        values = cli_values
    elif 'lr_methods' in sweep_section:
        values = sweep_section['lr_methods']
    else:
        print("WARNING: no lr_methods in the sweep config and no --lr_method: "
              "the LR of each model yaml is used (not in the checkpoint name)")
        values = [None]

    for value in values:
        if value is not None and value not in valid:
            fail(f"unknown LR method: {value}", valid)
    return values



    

def get_run_ids(sweep_section, cli_values=None):
    """Run ids to launch: the ones given with --run_id, or 1..runs of the sweep.
    Every id must be in 1..runs (training: one seed per run)."""
    valid = list(range(1, sweep_section['runs'] + 1))

    if cli_values is None:
        values = valid
    else:
        values = cli_values

    for value in values:
        if value not in valid:
            fail(f"unknown run id: {value}", valid)
    return values
