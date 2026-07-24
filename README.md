# SAM-Augmented Ensemble (Refactoring)

## 📂 Original Project

https://github.com/LorisNanni/Exploring-SAM-Augmented-Ensembles

This repository contains a framework for augmenting medical image datasets using Segment Anything Model (SAM/SAM2) and training segmentation models (HSNet, PolypPVT).

**The project is currently being restructured** to improve modularity, readability, and code maintainability. The final structure and some components may be subject to changes.

## 📋 Project Structure

```
├── configs/                    # YAML configuration files
│   ├── sam_augmentation.yaml  # SAM augmentation config
│   ├── da1_augmentation.yaml  # DA1 offline augmentation (paper-faithful)
│   ├── da2_augmentation.yaml  # DA2 offline augmentation (paper-faithful, 13 methods)
│   ├── hsnet_vanilla.yaml     # HSNet (no SAM augmentation)
│   ├── hsnet_aux.yaml         # HSNet (with SAM augmentation)
│   ├── polypPVT_vanilla.yaml  # PolypPVT (no SAM augmentation)
│   ├── polypPVT_aux.yaml      # PolypPVT (with SAM augmentation)
│   ├── sweep_train.yaml       # Training sweep (models + training axes)
│   ├── sweep_test.yaml        # Inference sweep (models + testing axes)
│   └── ensemble.yaml          # Ensemble evaluation paths
├── datasets/                  # Original datasets
├── output/                    # Generated outputs (gitignored)
│   ├── augmentation/          # Augmented datasets
│   ├── models/                # Saved model checkpoints
│   ├── predictions/           # Inference outputs
│   ├── logits/                # Raw pre-sigmoid logits (.npy)
│   └── ensemble/              # Ensemble outputs
├── scripts/                   # Executable scripts
│   ├── run_sam_augmentation.py  # SAM augmentation runner (was run_augmentation.py)
│   ├── run_da1_augmentation.py  # DA1 offline augmentation runner
│   ├── run_da2_augmentation.py  # DA2 offline augmentation runner (needs torchstain)
│   ├── run_training.py        # Training sweep runner
│   ├── run_inference.py       # Inference sweep runner 
│   └── run_ensemble.py        # Ensemble evaluation 
├── slurm/                     # SLURM job submission scripts
├── src/                      # Modular source code
│   ├── augmentation/         # Augmentation modules
│   │   ├── methods.py       # SAM augmentation methods (RG_logits, PCA_segPrior, ...)
│   │   ├── sam_loader.py    # SAM model loading
│   │   ├── da1_methods.py   # DA1 offline (fliplr, flipud, rot90 + foreground filter)
│   │   ├── da2_methods.py   # DA2 offline (13 methods: geometric + photometric + stain norm)
│   │   └── da3.py           # DA3 online (used inside dataloader)
│   ├── ensemble/             # Ensemble module
│   │   └── ensemble.py      # Mean rule ensemble + Dice evaluation
│   └── models/              # Segmentation models
│       ├── HSNet/           # HSNet model (vanilla + SAM-augmented)
│       │   ├── Train_vanilla.py  # Training (no SAM augmentation)
│       │   ├── Train_aux.py      # Training (with SAM augmentation)
│       │   ├── Test_vanilla.py   # Inference (no SAM augmentation)
│       │   ├── Test_aux.py       # Inference (with SAM augmentation, computes Dice)
│       │   ├── lib/         # Model libraries
│       │   ├── utils/       # Utilities
│       │   └── pretrained_pth/  # Pretrained weights
│       └── PolypPVT/        # PolypPVT model 
│           ├── Train_vanilla.py  # Training (no SAM augmentation)
│           ├── Train_aux.py      # Training (with SAM augmentation)
│           ├── Test_vanilla.py   # Inference (no SAM augmentation)
│           ├── Test_aux.py       # Inference (with SAM augmentation)
│           ├── lib/
│           └── utils/
├── checkpoints_sam/          # SAM checkpoints (downloaded automatically)
├── segment-anything-2/       # SAM2 repository (cloned automatically)
├── requirements.txt          # Python dependencies
└── setup.sh                  # Automatic setup script
```

## 🚀 Initial Setup

### 1. Installation

Run the automatic setup script which:
- Creates the Python 3.11 virtual environment
- Clones the SAM2 repository
- Installs all dependencies
- Downloads SAM1 and SAM2 checkpoints

```bash
bash setup.sh
```

### 2. Environment Activation

```bash
source venv_newSAMAug/bin/activate
```

## 📊 Usage

### Dataset Augmentation

Three families of augmentation are provided:

- **SAM augmentation** — [`scripts/run_sam_augmentation.py`](scripts/run_sam_augmentation.py) — uses SAM/SAM2 priors to augment images. Configured via [`configs/sam_augmentation.yaml`](configs/sam_augmentation.yaml).
- **DA1 offline** — [`scripts/run_da1_augmentation.py`](scripts/run_da1_augmentation.py) — 3 geometric methods (fliplr, flipud, rot90) + `<100 px` foreground filter. Produces a 4× dataset. Configured via [`configs/da1_augmentation.yaml`](configs/da1_augmentation.yaml).
- **DA2 offline** — [`scripts/run_da2_augmentation.py`](scripts/run_da2_augmentation.py) — 13 methods (5 geometric, 5 photometric, 3 stain normalization). Requires `torchstain` for Macenko. Configured via [`configs/da2_augmentation.yaml`](configs/da2_augmentation.yaml).

**Execution examples:**

```bash
python scripts/run_sam_augmentation.py --config configs/sam_augmentation.yaml
python scripts/run_da1_augmentation.py --config configs/da1_augmentation.yaml
python scripts/run_da2_augmentation.py --config configs/da2_augmentation.yaml
```

**Output:** Augmented datasets are saved under `output/augmentation/{sam|da1|da2}/<source>/<dataset>/{images,masks}/`

### HSNet Training

**Scripts:** [`src/models/HSNet/Train_vanilla.py`](src/models/HSNet/Train_vanilla.py) / [`Train_aux.py`](src/models/HSNet/Train_aux.py)

**Configuration:** [`configs/hsnet_vanilla.yaml`](configs/hsnet_vanilla.yaml) / [`configs/hsnet_aux.yaml`](configs/hsnet_aux.yaml)

For `Train_aux.py`, set `paths.aux_root_base` to the folder containing SAM-augmented images.

**Pretrained weights:** Download the HSNet pretrained model from [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV) and place it in `./src/models/HSNet/pretrained_pth` as in the original HSNet repository: https://github.com/baiboat/HSNet .

**Execution:**

```bash
python src/models/HSNet/Train_vanilla.py --config configs/hsnet_vanilla.yaml
python src/models/HSNet/Train_aux.py --config configs/hsnet_aux.yaml
# debug mode (1 epoch, 5 batches):
python src/models/HSNet/Train_vanilla.py --debug
python src/models/HSNet/Train_aux.py --debug
```

**Output:** Model checkpoints are saved in `output/models/`

### HSNet Inference

**Scripts:** [`src/models/HSNet/Test_vanilla.py`](src/models/HSNet/Test_vanilla.py) / [`Test_aux.py`](src/models/HSNet/Test_aux.py)

**Configuration:** Edit the `testing` and `datasets.test` sections in the respective YAML config.

**Execution:**

```bash
python src/models/HSNet/Test_vanilla.py --config configs/hsnet_vanilla.yaml
python src/models/HSNet/Test_aux.py --config configs/hsnet_aux.yaml
# or with explicit overrides:
python src/models/HSNet/Test_vanilla.py --model_pth output/models/HSNet_Baseline_DA3.pth --test_dataset TestDataset
python src/models/HSNet/Test_aux.py --model_pth output/models/HSNet_Aux_DA3.pth
```

**Output:** Predictions saved in `output/predictions/{dataset_name}/`. `Test_aux.py` also prints per-dataset Dice scores.

### PolypPVT Training 

**Scripts:** [`src/models/PolypPVT/Train_vanilla.py`](src/models/PolypPVT/Train_vanilla.py) / [`Train_aux.py`](src/models/PolypPVT/Train_aux.py)

**Configuration:** [`configs/polypPVT_vanilla.yaml`](configs/polypPVT_vanilla.yaml) / [`configs/polypPVT_aux.yaml`](configs/polypPVT_aux.yaml)

```bash
python src/models/PolypPVT/Train_vanilla.py --config configs/polypPVT_vanilla.yaml
python src/models/PolypPVT/Train_aux.py --config configs/polypPVT_aux.yaml
# debug mode:
python src/models/PolypPVT/Train_vanilla.py --debug
```

### PolypPVT Inference 

**Scripts:** [`src/models/PolypPVT/Test_vanilla.py`](src/models/PolypPVT/Test_vanilla.py) / [`Test_aux.py`](src/models/PolypPVT/Test_aux.py)

```bash
python src/models/PolypPVT/Test_vanilla.py --config configs/polypPVT_vanilla.yaml
python src/models/PolypPVT/Test_aux.py --config configs/polypPVT_aux.yaml
```

**Output:** Predictions in `output/predictions/{model_name}/{dataset}/`, logits in `output/logits/{model_name}/{dataset}/`

### 🔄 Training Sweep Runner ⚠️ not yet cluster-validated

**Script to run:** [`scripts/run_training.py`](scripts/run_training.py)

**Configuration:** Edit [`configs/sweep_train.yaml`](configs/sweep_train.yaml) to select models, methods, SAM versions and number of runs. Used by default (no `--sweep` needed).

**Execution:**

```bash
# Run all models, all combinations (defaults to configs/sweep_train.yaml)
python scripts/run_training.py

# Filter to one model only
python scripts/run_training.py --model hsnet_aux

# Run a subset of runs (nargs: several IDs at once)
python scripts/run_training.py --model hsnet_aux --run_id 2 3 4

# SLURM array job (one run per job; --run_id gets a single value)
python scripts/run_training.py --model hsnet_aux --run_id $SLURM_ARRAY_TASK_ID
```

Runs whose checkpoint already exists are skipped automatically; pass `--force` to retrain and overwrite them.

**Output:** Checkpoints saved as `output/models/sam{v}_{method}_{model_name}_{da}_{lr}_run{id}.pth`

### 🔄 Inference Sweep Runner

**Script to run:** [`scripts/run_inference.py`](scripts/run_inference.py)

**Configuration:** Uses [`configs/sweep_test.yaml`](configs/sweep_test.yaml) (its `testing` section) — a separate file from `sweep_train.yaml`, so a test sweep can be edited while a training job is still queued. Used by default (no `--sweep` needed).

**Execution:**

```bash
python scripts/run_inference.py
python scripts/run_inference.py --model hsnet_aux
python scripts/run_inference.py --model hsnet_aux --run_id 3
python scripts/run_inference.py --model hsnet_aux --run_id 2 3 4
```

**Output:** Predictions saved in `output/predictions/{model_name}/{dataset}/`

### 🔄 Ensemble Evaluation

**Script to run:** [`scripts/run_ensemble.py`](scripts/run_ensemble.py)

**Configuration:** [`configs/ensemble.yaml`](configs/ensemble.yaml) — defines default paths:
- `paths.models_outputs` — folder containing one subfolder per model (i.e. `output/predictions/`)
- `paths.test_masks` — datasets root, used to resolve `{dataset}/masks/`
- `paths.out_folder` — where averaged predictions are saved

**Execution:**

```bash
# Use defaults from configs/ensemble.yaml
python scripts/run_ensemble.py

# Override paths via CLI
python scripts/run_ensemble.py --models_outputs output/predictions/ --out_folder output/ensemble/
```

Loads all model prediction subfolders under `models_outputs`, averages predictions pixel-wise (mean rule after sigmoid), prints per-dataset Dice and overall mean.

### ⚙️ Sweep Configuration (`configs/sweep_train.yaml`, `configs/sweep_test.yaml`)

Experiments are driven by two files, one per stage:
- [`configs/sweep_train.yaml`](configs/sweep_train.yaml) — `models` + `training` axes (used by `run_training.py`)
- [`configs/sweep_test.yaml`](configs/sweep_test.yaml) — `models` + `testing` axes (used by `run_inference.py`)

The `models` list is duplicated across both files on purpose and must stay in sync
(same names, folders, scripts, config path).

Key fields:
- `models` — which models to run: their folder, scripts, per-model `config` yaml, and whether they use SAM augmentation (`has_aux: true/false`)
- `training.seeds` — one seed per run for reproducibility across the 5 runs
- `sam_versions` — `[1]`, `[2]`, or `[1, 2]`
- `aug_methods` — comment/uncomment to select methods
- `da_methods` / `lr_methods` — DA and LR axes; `sweep_test.da_methods` must match `sweep_train.da_methods` to find the checkpoints

If `has_aux: true` → runner loops over `sam_versions × aug_methods × da_methods × lr_methods × runs`.
If `has_aux: false` → runner loops over `da_methods × lr_methods × runs`.

Checkpoint naming (`build_model_name`) and combo validity (`is_valid_combo`) live in
[`src/sweep/naming.py`](src/sweep/naming.py), shared by both runners so the training
and testing stages can never disagree on filenames.

For SLURM: submit with `--array=1-5` and pass `$SLURM_ARRAY_TASK_ID` as `--run_id`. Each array job runs all method combinations for that run ID in parallel with the others.

## 🔧 Available Augmentation Methods

### SAM-based methods

Implemented in [`src/augmentation/methods.py`](src/augmentation/methods.py):

- `SAMAug` - SAM segmentation prior added to G and B channels
- `ourSAMAug` - Custom SAM augmentation
- `RG_segPrior` - Random Gaussian with segmentation prior
- `SV_segPrior` - HSV Color Space with Segmentation Prior (H-Channel Encoding)
- `RG_logits` - Random Gaussian based on logits
- `PCA_segPrior` - PCA with segmentation prior

### Offline DA methods (paper *An empirical study on ensemble of segmentation approaches*)

Ported 1:1 from the MATLAB reference toolbox. Configuration and runners live in `configs/` and `scripts/`, functions in `src/augmentation/`.

- **DA1** — [`src/augmentation/da1_methods.py`](src/augmentation/da1_methods.py)
  - `fliplr`, `flipud`, `rot90` (deterministic) + `has_enough_foreground` filter
  - Produces a 4× dataset (original + 3 variants per image)

- **DA2** — [`src/augmentation/da2_methods.py`](src/augmentation/da2_methods.py) — 13 methods:
  - Geometric (5): `width_shift`, `height_shift`, `rotation`, `shear`, `random_flip`
  - Photometric (5): `brightness_uniform`, `brightness_per_channel`, `speckle_noise`, `contrast_blur`, `shadows`
  - Stain normalization (3): `rgb_histogram_match`, `reinhard_normalize`, `macenko_normalize` (via `torchstain`)

- **DA3** — [`src/augmentation/da3.py`](src/augmentation/da3.py)
  - Online augmentation applied inside the model dataloader (rotation + flip + color jitter). Not run standalone; enabled per model via the model config yaml.

## 📦 Main Dependencies

- **Python 3.11**
- **PyTorch** (installed with SAM2)
- **segment-anything** (SAM v1)
- **segment-anything-2** (SAM v2)
- **timm** - For model backbones
- **opencv-python, scikit-image, scipy** - Image processing (used by DA2)
- **torchstain** - Macenko stain normalization for DA2 (`pip install torchstain`)
- **PyYAML** - Configuration management

## 📝 Notes

- SAM checkpoints are automatically downloaded to `checkpoints_sam/`
- Dataset structure must follow the format: `{dataset}/images/` and `{dataset}/masks/`
- All generated outputs (models, predictions, augmented data) are saved in `output/` and are gitignored
- All 4 Test scripts save raw logits (pre-sigmoid, float32, `.npy`) in `output/logits/`
- PolypPVT scripts are complete but not yet validated on the cluster
- Runner scripts have not yet been validated on the cluster

---

*Last updated: 03 Luglio 2026*
