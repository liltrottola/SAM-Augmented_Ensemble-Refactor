# SAM-Augmented Ensemble (Refactoring)

⚠️ Runner scripts (`run_training.py`, `run_inference.py`, `run_ensemble.py`) not yet validated on the cluster.

⚠️ PolypPVT training and inference not yet validated on the cluster.

## 📂 Original Project

https://github.com/LorisNanni/Exploring-SAM-Augmented-Ensembles

This repository contains a framework for augmenting medical image datasets using Segment Anything Model (SAM/SAM2) and training segmentation models (HSNet, PolypPVT).

**The project is currently being restructured** to improve modularity, readability, and code maintainability. The final structure and some components may be subject to changes.

## 📋 Project Structure

```
├── configs/                    # YAML configuration files
│   ├── augmentation.yaml      # Configuration for augmentation
│   ├── hsnet_vanilla.yaml     # HSNet (no SAM augmentation)
│   ├── hsnet_aux.yaml         # HSNet (with SAM augmentation)
│   ├── polypPVT_vanilla.yaml  # PolypPVT (no SAM augmentation)
│   ├── polypPVT_aux.yaml      # PolypPVT (with SAM augmentation)
│   └── sweep.yaml             # Experiment sweep (training + inference)
├── datasets/                  # Original datasets
├── output/                    # Generated outputs (gitignored)
│   ├── augmentation/          # Augmented datasets
│   ├── models/                # Saved model checkpoints
│   ├── predictions/           # Inference outputs
│   ├── logits/                # Raw pre-sigmoid logits (.npy)
│   └── ensemble/              # Ensemble outputs
├── scripts/                   # Executable scripts
│   ├── run_augmentation.py    # SAM augmentation runner
│   ├── run_training.py        # Training sweep runner ⚠️ not yet validated
│   ├── run_inference.py       # Inference sweep runner ⚠️ not yet validated
│   └── run_ensemble.py        # Ensemble evaluation ⚠️ not yet validated
├── slurm/                     # SLURM job submission scripts
├── src/                      # Modular source code
│   ├── augmentation/         # Augmentation modules
│   │   ├── methods.py       # Augmentation methods
│   │   └── sam_loader.py    # SAM model loading
│   ├── ensemble/             # Ensemble module
│   │   └── ensemble.py      # Mean rule ensemble + Dice evaluation
│   └── models/              # Segmentation models
│       ├── HSNet/           # HSNet model
│       │   ├── Train.py     # Training script
│       │   ├── Test.py      # Testing/inference script
│       │   ├── lib/         # Model libraries
│       │   ├── utils/       # Utilities
│       │   └── pretrained_pth/  # Pretrained weights
│       ├── HSNet_aux/       # HSNet with SAM-augmented images
│       │   ├── Train.py     # Training script
│       │   ├── Test.py      # Testing/inference script (computes Dice)
│       │   ├── lib/         # Model libraries
│       │   ├── utils/dataloader.py
│       │   └── pretrained_pth/
│       └── PolypPVT/        # PolypPVT model ⚠️ not yet cluster-validated
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

**Script to run:** [`scripts/run_augmentation.py`](scripts/run_augmentation.py)

**Configuration:** Edit the [`configs/augmentation.yaml`](configs/augmentation.yaml) file to specify:
- Datasets to process (`datasets.folders`)
- Augmentation methods (`augmentation.methods`)
- SAM versions to use (`sam.versions`)
- Checkpoint paths (`paths.checkpoints_root`)

**Execution:**

```bash
python scripts/run_augmentation.py --config configs/augmentation.yaml
```

**Output:** Augmented datasets are saved in `output/augmentation/`

### HSNet Training

**Script to run:** [`src/models/HSNet/Train.py`](src/models/HSNet/Train.py)

**Configuration:** Edit the [`configs/hsnet_vanilla.yaml`](configs/hsnet_vanilla.yaml) file to specify training parameters.

**Pretrained weights:** Download the HSNet pretrained model from [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV) and place it in `./src/models/HSNet/pretrained_pth` as in the original HSNet repository: https://github.com/baiboat/HSNet .

**Execution:**

```bash
python src/models/HSNet/Train.py
# or with explicit config:
python src/models/HSNet/Train.py --config configs/hsnet_vanilla.yaml
# debug mode (1 epoch, 5 batches):
python src/models/HSNet/Train.py --debug
```

**Output:** Model checkpoints are saved in `output/models/`

### HSNet Aux Training (SAM-augmented)

**Script to run:** [`src/models/HSNet_aux/Train.py`](src/models/HSNet_aux/Train.py)

**Configuration:** Edit [`configs/hsnet_aux.yaml`](configs/hsnet_aux.yaml). Set `paths.aux_root_base` to the folder containing SAM-augmented images.

**Execution:**

```bash
python src/models/HSNet_aux/Train.py
# or with explicit config:
python src/models/HSNet_aux/Train.py --config configs/hsnet_aux.yaml
# debug mode (5 batches):
python src/models/HSNet_aux/Train.py --debug
```

**Output:** Model checkpoints are saved in `output/models/`

### HSNet Inference

**Script to run:** [`src/models/HSNet/Test.py`](src/models/HSNet/Test.py)

**Configuration:** Edit the `testing` and `datasets.test` sections in [`configs/hsnet_vanilla.yaml`](configs/hsnet_vanilla.yaml).

**Execution:**

```bash
python src/models/HSNet/Test.py
# or with explicit overrides:
python src/models/HSNet/Test.py --model_pth output/models/HSNet_Baseline_DA3.pth --test_dataset TestDataset
```

**Output:** Predictions are saved in `output/predictions/{dataset_name}/`

### HSNet Aux Inference

**Script to run:** [`src/models/HSNet_aux/Test.py`](src/models/HSNet_aux/Test.py)

**Configuration:** Edit the `testing` and `datasets.test` sections in [`configs/hsnet_aux.yaml`](configs/hsnet_aux.yaml).

**Execution:**

```bash
python src/models/HSNet_aux/Test.py
# or with explicit overrides:
python src/models/HSNet_aux/Test.py --model_pth output/models/HSNet_Aux_DA3.pth
```

**Output:** Per-dataset Dice scores printed to console. Predictions saved in `output/predictions/{dataset_name}/`

### PolypPVT Training ⚠️ not yet cluster-validated

**Scripts:** [`src/models/PolypPVT/Train_vanilla.py`](src/models/PolypPVT/Train_vanilla.py) / [`Train_aux.py`](src/models/PolypPVT/Train_aux.py)

**Configuration:** [`configs/polypPVT_vanilla.yaml`](configs/polypPVT_vanilla.yaml) / [`configs/polypPVT_aux.yaml`](configs/polypPVT_aux.yaml)

```bash
python src/models/PolypPVT/Train_vanilla.py --config configs/polypPVT_vanilla.yaml
python src/models/PolypPVT/Train_aux.py --config configs/polypPVT_aux.yaml
# debug mode:
python src/models/PolypPVT/Train_vanilla.py --debug
```

### PolypPVT Inference ⚠️ not yet cluster-validated

**Scripts:** [`src/models/PolypPVT/Test_vanilla.py`](src/models/PolypPVT/Test_vanilla.py) / [`Test_aux.py`](src/models/PolypPVT/Test_aux.py)

```bash
python src/models/PolypPVT/Test_vanilla.py --config configs/polypPVT_vanilla.yaml
python src/models/PolypPVT/Test_aux.py --config configs/polypPVT_aux.yaml
```

**Output:** Predictions in `output/predictions/{model_name}/{dataset}/`, logits in `output/logits/{model_name}/{dataset}/`

### 🔄 Training Sweep Runner ⚠️ not yet cluster-validated

**Script to run:** [`scripts/run_training.py`](scripts/run_training.py)

**Configuration:** Edit [`configs/sweep.yaml`](configs/sweep.yaml) to select models, methods, SAM versions and number of runs.

**Execution:**

```bash
# Run all models, all combinations
python scripts/run_training.py --sweep configs/sweep.yaml

# Filter to one model only
python scripts/run_training.py --sweep configs/sweep.yaml --model hsnet_aux

# SLURM array job (one run per job)
python scripts/run_training.py --sweep configs/sweep.yaml --model hsnet_aux --run_id $SLURM_ARRAY_TASK_ID
```

**Output:** Checkpoints saved as `output/models/sam{v}_{method}_{model_name}_run{id}.pth`

### 🔄 Inference Sweep Runner ⚠️ not yet cluster-validated

**Script to run:** [`scripts/run_inference.py`](scripts/run_inference.py)

**Configuration:** Uses the `testing` section of [`configs/sweep.yaml`](configs/sweep.yaml) — independent from `training`, so inference can target a different subset.

**Execution:**

```bash
python scripts/run_inference.py --sweep configs/sweep.yaml
python scripts/run_inference.py --sweep configs/sweep.yaml --model hsnet_aux
python scripts/run_inference.py --sweep configs/sweep.yaml --model hsnet_aux --run_id 3
```

**Output:** Predictions saved in `output/predictions/{model_name}/{dataset}/`

### 🔄 Ensemble Evaluation ⚠️ not yet cluster-validated

**Script to run:** [`scripts/run_ensemble.py`](scripts/run_ensemble.py)

**Execution:**

```bash
python scripts/run_ensemble.py \
    --models_outputs output/predictions/ \
    --test_masks datasets/TestDatasets/TestDataset/ \
    --out_folder output/ensemble/
```

Loads all model prediction subfolders under `--models_outputs`, averages predictions pixel-wise (mean rule after sigmoid, threshold 0.5), prints per-dataset Dice and overall mean.

### ⚙️ Sweep Configuration (`configs/sweep.yaml`)

`sweep.yaml` is the central configuration point for running experiments. It has two independent sections — `training` and `testing`.

Key fields:
- `models` — which models to run, their folder, scripts, and whether they use SAM augmentation (`has_aux: true/false`)
- `training.seeds` — one seed per run for reproducibility across the 5 runs
- `training.sam_versions` / `testing.sam_versions` — `[1]`, `[2]`, or `[1, 2]`
- `training.aug_methods` / `testing.aug_methods` — comment/uncomment to select methods

If `has_aux: true` → runner loops over `sam_versions × aug_methods × runs`.
If `has_aux: false` → runner loops over `runs` only.

For SLURM: submit with `--array=1-5` and pass `$SLURM_ARRAY_TASK_ID` as `--run_id`. Each array job runs all method combinations for that run ID in parallel with the others.

## 🔧 Available Augmentation Methods

Methods implemented in [`src/augmentation/methods.py`](src/augmentation/methods.py):

- `SAMAug` - SAM segmentation prior added to G and B channels
- `ourSAMAug` - Custom SAM augmentation
- `RG_segPrior` - Random Gaussian with segmentation prior
- `SV_segPrior` - HSV Color Space with Segmentation Prior (H-Channel Encoding)
- `RG_logits` - Random Gaussian based on logits
- `PCA_segPrior` - PCA with segmentation prior

## 📦 Main Dependencies

- **Python 3.11**
- **PyTorch** (installed with SAM2)
- **segment-anything** (SAM v1)
- **segment-anything-2** (SAM v2)
- **timm** - For model backbones
- **opencv-python, scikit-image** - Image processing
- **PyYAML** - Configuration management

## 📝 Notes

- SAM checkpoints are automatically downloaded to `checkpoints_sam/`
- Dataset structure must follow the format: `{dataset}/images/` and `{dataset}/masks/`
- All generated outputs (models, predictions, augmented data) are saved in `output/` and are gitignored
- All 4 Test scripts save raw logits (pre-sigmoid, float32, `.npy`) in `output/logits/`
- PolypPVT scripts are complete but not yet validated on the cluster
- Runner scripts have not yet been validated on the cluster

---

*Last updated: 31 March 2026*
