# sweep/

Shared logic that drives parameter sweeps across the four model variants
(hsnet, hsnet_aux, polypvt, polypvt_aux).

## Why this package exists

`scripts/run_training.py` and `scripts/run_inference.py` must agree on two things,
or the test stage looks for a checkpoint the training stage never wrote:

1. the checkpoint filename        -> `build_model_name`
2. which (aux_source, method) pairs to sweep for aux models -> `get_aux_sources`
3. which (model, DA) combos are valid -> `is_valid_combo`

Both live in `naming.py` so the two runners import the same code and cannot drift.

## Files

- `naming.py` — checkpoint naming + aux axis + combo validity (single source of truth).

## Config layout

- `configs/sweep_train.yaml` — `models` + `training` axes (read by `run_training.py`).
- `configs/sweep_test.yaml`  — `models` + `testing` axes (read by `run_inference.py`).

The `models` list is duplicated across the two sweep files on purpose: it lets a
test sweep be edited while a training job is still queued, without both stages
sharing one file. Keep the two lists in sync (names, folders, scripts, config).

## Aux axis

Aux models sweep over `aux_sources`, a map from source folder (under
`output/augmentation/`) to the methods that exist under it:

```yaml
aux_sources:
  sam1: ["SAMAug", "RG_logits", ...]
  sam2: ["SAMAug", "RG_logits", ...]
  da3mono-large: ["DepthAug", "RG_voteV3", ...]
```

A map because SAM methods and depth methods only
exist under their own source. The runners pass `--aux_source <source> --method <method>`
to `Train_aux.py` / `Test_aux.py`. A new aux source needs no code: generate
`output/augmentation/<source>/<method>/<dataset>/images/` (PNG, same file names as
the dataset images, for TrainDataset and every test set) and add one line here.

## Checkpoint naming

`build_model_name` produces, e.g.:

- vanilla: `hsnet_da3_lrbase_run1`
- aux:     `sam1_ourSAMAug_polypvt_aux_da3_lrbase_run1`
- depth:   `da3mono-large_DepthAug_hsnet_aux_da3_lrbase_run1`

The aux prefix is the source folder name

`run_id` is appended last, so runs of the same experiment sort together on disk.
The Test scripts read the whole stem back as the output folder name; nothing
parses the individual fields out, so the field order is free to change here.

## Resume / skip (training)

`run_training.py` skips any run whose checkpoint already exists (path resolved from
each model's `models_dir` in its per-model yaml). Use `--force` to retrain and
overwrite. Example: run 1 already trained -> only the missing runs launch.

Safe pattern for `--force`: always scope it, e.g. `--force --model hsnet --run_id 3`,
never on the whole sweep (it overwrites checkpoints with no backup).

## Selecting runs

`--run_id` takes one or more IDs:

- `--run_id 3`       -> just run 3
- `--run_id 2 3 4`   -> runs 2, 3, 4
- omitted            -> all runs (1..runs)
- SLURM: `--run_id $SLURM_ARRAY_TASK_ID` (a single value per array task)

`--aux_source` takes one or more keys of `aux_sources` (aux models only):

- `--aux_source da3mono-large`          -> only the depth aux
- `--aux_source sam2_stab085 da3mono-large` -> those two sources
- omitted                              -> all the sources in the sweep yaml
- a name not in `aux_sources`          -> error with the list of the available ones

## TODO / future work

- [ ] DA registry: replace the `OFFLINE_DA_METHODS` set in `src/augmentation/__init__.py`
      with per-method data (`modality: offline|online`, `aux_compatible or not`) once
      more DA methods land, so runners stop hardcoding the rule.
- [ ] extract a full RunSpec / generate_run_specs generator if the nested loops grow
- [ ] freeze the sweep yaml and the per-model yaml too (runner passes --sweep and --config snapshot to Train_*.py)
