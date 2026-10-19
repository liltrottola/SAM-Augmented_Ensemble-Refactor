# SAM-Augmented Ensemble (Refactoring)

## 📂 Original Project

https://github.com/LorisNanni/Exploring-SAM-Augmented-Ensembles

This repository contains a framework for augmenting medical image datasets using Segment Anything Model (SAM/SAM2), Depth Anything model and training segmentation models (HSNet, PolypPVT; HGM and SAM2-UNet as vanilla models).

**The project is currently being restructured** to improve modularity, readability, and code maintainability. The final structure and some components may be subject to changes.

## 📋 Project Structure

```
├── configs/                    # YAML configuration files
│   ├── sam_generation.yaml    # SAM priors generation (runs SAM)
│   ├── sam_augmentation.yaml  # SAM aux images from the saved SAM priors
│   ├── depth_prior_maps.yaml  # Depth priors from the DA3 depth
│   ├── depth_augmentation.yaml # Depth aux images from the saved depth priors
│   ├── da1_augmentation.yaml  # DA1 offline augmentation (paper-faithful)
│   ├── da2_augmentation.yaml  # DA2 offline augmentation (paper-faithful, 13 methods)
│   ├── hsnet_vanilla.yaml     # HSNet (no SAM augmentation)
│   ├── hsnet_aux.yaml         # HSNet (with SAM augmentation)
│   ├── polypPVT_vanilla.yaml  # PolypPVT (no SAM augmentation)
│   ├── polypPVT_aux.yaml      # PolypPVT (with SAM augmentation)
│   ├── hgm_vanilla.yaml       # HGM (Hybrid Graph Mamba), vanilla only
│   ├── sam2unet_vanilla.yaml  # SAM2-UNet, vanilla only
│   ├── sweep_train.yaml       # Training sweep (models + training axes)
│   ├── sweep_test.yaml        # Inference sweep (models + testing axes)
│   └── ensemble.yaml          # Ensemble evaluation paths
├── datasets/                  # Original datasets
├── output/                    # Generated outputs (gitignored)
│   ├── priors/                # Prior maps (.npy): <source>/<dataset>/<prior>/<name>.npy
│   ├── augmentation/          # Augmented datasets / aux images
│   ├── models/                # Saved model checkpoints
│   ├── predictions/           # Inference outputs
│   ├── logits/                # Raw pre-sigmoid logits (.npy)
│   └── ensemble/              # Ensemble outputs
├── scripts/                   # Executable scripts
│   ├── run_sam_generation.py    # SAM priors (segPrior, boundaryPrior, areaPrior, logits)
│   ├── run_sam_augmentation.py  # SAM aux images from the saved SAM priors
│   ├── run_depth_generation.py  # Raw depth with Depth Anything 3
│   ├── run_depth_prior_maps.py  # Depth priors (frst_prior, localnorm, depth_p1p99)
│   ├── run_depth_augmentation.py # Depth aux images from the saved depth priors
│   ├── run_da1_augmentation.py  # DA1 offline augmentation runner
│   ├── run_da2_augmentation.py  # DA2 offline augmentation runner (needs torchstain)
│   ├── run_training.py        # Training sweep runner
│   ├── run_inference.py       # Inference sweep runner 
│   └── run_ensemble.py        # Ensemble evaluation 
├── slurm/                     # SLURM job submission scripts
├── src/                      # Modular source code
│   ├── augmentation/         # Augmentation modules
│   │   ├── sam_methods.py   # SAM priors generation (+ methods that run SAM inside)
│   │   ├── sam_prior_methods.py   # SAM aux methods from the saved priors (SAMAug, RG_logits, ...)
│   │   ├── depth_prior_methods.py # Depth priors + depth aux methods (DepthAug, RG_voteV3, ...)
│   │   ├── sam_loader.py    # SAM model loading
│   │   ├── da1_methods.py   # DA1 offline (fliplr, flipud, rot90 + foreground filter)
│   │   ├── da2_methods.py   # DA2 offline (13 methods: geometric + photometric + stain norm)
│   │   └── da3.py           # DA3 online (used inside dataloader)
│   ├── ensemble/             # Ensemble module
│   │   └── ensemble.py      # Mean rule ensemble + Dice evaluation
│   ├── sweep/                # Shared sweep logic (checkpoint naming, aux sources)
│   └── models/              # Segmentation models
│       ├── HSNet/           # HSNet model (vanilla + SAM-augmented)
│       │   ├── Train_vanilla.py  # Training (no SAM augmentation)
│       │   ├── Train_aux.py      # Training (with aux images)
│       │   ├── Test_vanilla.py   # Inference (no SAM augmentation)
│       │   ├── Test_aux.py       # Inference (with aux images, computes Dice)
│       │   ├── lib/         # Model libraries
│       │   ├── utils/       # Utilities
│       │   └── pretrained_pth/  # Pretrained weights
│       ├── PolypPVT/        # PolypPVT model 
│       │   ├── Train_vanilla.py  # Training 
│       │   ├── Train_aux.py      # Training (with aux images)
│       │   ├── Test_vanilla.py   # Inference 
│       │   ├── Test_aux.py       # Inference (with aux images)
│       │   ├── lib/
│       │   └── utils/
│       ├── HGM/             # HGM model, vanilla only (own conda env, see its README)
│       │   ├── Train_vanilla.py
│       │   ├── Test_vanilla.py
│       │   ├── lib/  utils/  pretrained_pth/
│       │   └── README.md
│       └── SAM2UNet/        # SAM2-UNet model, vanilla only
│           ├── Train_vanilla.py
│           ├── Test_vanilla.py
│           ├── SAM2UNet.py
│           ├── sam2/  sam2_configs/   # SAM2 bundled with SAM2-UNet
│           └── README.md
├── checkpoints_sam/          # SAM checkpoints (downloaded automatically)
├── segment-anything-2/       # SAM2 repository (cloned automatically)
├── requirements.txt          # Python dependencies
├── requirements_hgm.txt      # Dependencies of the HGM conda env
├── setup.sh                  # Automatic setup script
└── setup_HGM.sh              # Setup of the HGM conda env
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

#### Aux images (SAM and depth): priors first, then aux images

The aux images used by `Train_aux.py` / `Test_aux.py` are built in two steps. The model (e.g. SAM or Depth Anything 3) runs once and its **prior maps** are saved as `.npy`; the aux images are then built from those maps, without running the model again. 

| Step | SAM | Depth |
|---|---|---|
| 0. raw model output | — | [`run_depth_generation.py`](scripts/run_depth_generation.py) → `output/depth/da3mono-large/` |
| 1. priors (`.npy`) | [`run_sam_generation.py`](scripts/run_sam_generation.py) ([`sam_generation.yaml`](configs/sam_generation.yaml)), GPU | [`run_depth_prior_maps.py`](scripts/run_depth_prior_maps.py) ([`depth_prior_maps.yaml`](configs/depth_prior_maps.yaml)) |
| | → `output/priors/sam2_stab085/<dataset>/{segPrior,boundaryPrior,areaPrior,logits}/` | → `output/priors/da3mono-large/<dataset>/{frst_prior,localnorm,depth_p1p99}/` |
| 2. aux images (PNG) | [`run_sam_augmentation.py`](scripts/run_sam_augmentation.py) ([`sam_augmentation.yaml`](configs/sam_augmentation.yaml)), CPU | [`run_depth_augmentation.py`](scripts/run_depth_augmentation.py) ([`depth_augmentation.yaml`](configs/depth_augmentation.yaml)), CPU |
| | → `output/augmentation/<aux_source>/<method>/<dataset>/images/` | → `output/augmentation/da3mono-large/<method>/<dataset>/images/` |

```bash
# SAM
python scripts/run_sam_generation.py   --config configs/sam_generation.yaml
python scripts/run_sam_augmentation.py --config configs/sam_augmentation.yaml
# depth
python scripts/run_depth_generation.py
python scripts/run_depth_prior_maps.py   --config configs/depth_prior_maps.yaml
python scripts/run_depth_augmentation.py --config configs/depth_augmentation.yaml
# quick test on the first N images of each dataset (prior maps and aux images)
python scripts/run_depth_augmentation.py --limit 3
```

- Prior files are named as the image without extension (`149.png` → `149.npy`); aux images are always saved as PNG with the image name, since the aux loaders pair image and aux by sorted file name.
- The SAM prior folder is built from the SAM version and threshold: `sam1`, or `sam2_stab<threshold>` (e.g. `sam2_stab085`).
- In the aux yaml, `prior_source` = folder of the priors to read, `aux_source` = folder the aux images are written to (and the name used by the sweeps)..
- The aux methods are chosen by name in the yaml: each is a function `f(tI, priors) -> uint8 image` in [`sam_prior_methods.py`](src/augmentation/sam_prior_methods.py) or [`depth_prior_methods.py`](src/augmentation/depth_prior_methods.py), and its name is also the output folder. To add a method, write the function and add its name to the yaml.

#### Offline DA

- **DA1 offline** — [`scripts/run_da1_augmentation.py`](scripts/run_da1_augmentation.py) — 3 geometric methods (fliplr, flipud, rot90) + `<100 px` foreground filter. Produces a 4× dataset. Configured via [`configs/da1_augmentation.yaml`](configs/da1_augmentation.yaml).
- **DA2 offline** — [`scripts/run_da2_augmentation.py`](scripts/run_da2_augmentation.py) — 13 methods (5 geometric, 5 photometric, 3 stain normalization). Requires `torchstain` for Macenko. Configured via [`configs/da2_augmentation.yaml`](configs/da2_augmentation.yaml).

**Execution examples:**

```bash
python scripts/run_da1_augmentation.py --config configs/da1_augmentation.yaml
python scripts/run_da2_augmentation.py --config configs/da2_augmentation.yaml
```

**Output:** Augmented datasets are saved under `output/augmentation/{da1|da2}/<source>/<dataset>/{images,masks}/`

### HSNet Training

**Scripts:** [`src/models/HSNet/Train_vanilla.py`](src/models/HSNet/Train_vanilla.py) / [`Train_aux.py`](src/models/HSNet/Train_aux.py)

**Configuration:** [`configs/hsnet_vanilla.yaml`](configs/hsnet_vanilla.yaml) / [`configs/hsnet_aux.yaml`](configs/hsnet_aux.yaml)

`Train_aux.py` / `Test_aux.py` read the aux images from `<aux_root_base>/<aux_source>/<method>/<dataset>/images/`; set `aux_source` and `method` in the yaml or with `--aux_source` / `--method`.

**Pretrained weights:** Download the HSNet pretrained model from [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV) and place it in `./src/models/HSNet/pretrained_pth` as in the original HSNet repository: https://github.com/baiboat/HSNet .

**Execution:**

```bash
python src/models/HSNet/Train_vanilla.py --config configs/hsnet_vanilla.yaml
python src/models/HSNet/Train_aux.py --config configs/hsnet_aux.yaml
python src/models/HSNet/Train_aux.py --aux_source da3mono-large --method DepthAug
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

### HGM and SAM2-UNet
No aux version. Details and differences from their original code in their READMEs.

| | HGM | SAM2-UNet |
|---|---|---|
| Paper | *Hybrid Graph Mamba* (MICCAI 2025) | *SAM2-UNet* (2024) |
| Model | PVTv2-b2 + Mamba/GCN modules | frozen SAM2 Hiera-L encoder + adapters + U-Net decoder |
| Config | [`configs/hgm_vanilla.yaml`](configs/hgm_vanilla.yaml) | [`configs/sam2unet_vanilla.yaml`](configs/sam2unet_vanilla.yaml) |
| Environment | own conda env `HGM` (`mamba_ssm`, `torch_geometric`, torch 2.2.1 cu121) | main `venv_newSAMAug` |
| Pretrained weights | `src/models/HGM/pretrained_pth/pvt_v2_b2.pth` (same file as HSNet) | `checkpoints_sam/sam2_hiera_large.pt` (from `setup.sh`), training only |
| README | [`src/models/HGM/README.md`](src/models/HGM/README.md) | [`src/models/SAM2UNet/README.md`](src/models/SAM2UNet/README.md) |

**HGM environment setup** (once, on the cluster):

```bash
mkdir -p logs && sbatch slurm/run_setup_hgm.slurm   # runs setup_HGM.sh, creates the conda env "HGM"
conda activate HGM
```

**Execution** (from the model folder, as HSNet):

```bash
python Train_vanilla.py [--debug] [--model_name NAME] [--online_da da3|noda] [--offline_da da1|da2|noda] [--lr_method ...]
python Test_vanilla.py --model_pth NAME.pth
```

**With the sweep runners** (sweep entries `hgm` and `sam2unet`):

```bash
python scripts/run_training.py  --model hgm --run_id 1    # from the HGM conda env
python scripts/run_inference.py --model hgm --run_id 1
python scripts/run_training.py  --model sam2unet --run_id 1
python scripts/run_inference.py --model sam2unet --run_id 1
```

⚠️ HGM must always be launched with `--model hgm` from the `HGM` env: a full sweep (no `--model`) runs from `venv_newSAMAug` and would also try HGM, which does not run there.

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

# Only some aux sources (keys of aux_sources in the sweep yaml)
python scripts/run_training.py --model hsnet_aux --aux_source da3mono-large
```

Runs whose checkpoint already exists are skipped automatically; pass `--force` to retrain and overwrite them.

**Output:** Checkpoints saved as `output/models/{aux_source}_{method}_{model_name}_{da}_{lr}_run{id}.pth` for the aux models (e.g. `sam2_RG_logits_hsnet_aux_da3_lrbase_run1`), `{model_name}_{da}_{lr}_run{id}.pth` for the vanilla ones

### 🔄 Inference Sweep Runner

**Script to run:** [`scripts/run_inference.py`](scripts/run_inference.py)

**Configuration:** Uses [`configs/sweep_test.yaml`](configs/sweep_test.yaml) (its `testing` section) — a separate file from `sweep_train.yaml`, so a test sweep can be edited while a training job is still queued. Used by default (no `--sweep` needed).

**Execution:**

```bash
python scripts/run_inference.py
python scripts/run_inference.py --model hsnet_aux
python scripts/run_inference.py --model hsnet_aux --run_id 3
python scripts/run_inference.py --model hsnet_aux --run_id 2 3 4
python scripts/run_inference.py --model hsnet_aux --aux_source sam2_stab085 da3mono-large
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

# Name of this ensemble run (output in <out_folder>/<name>/)
python scripts/run_ensemble.py --name sam2_all_methods
```

Loads all model prediction subfolders under `models_outputs`, averages predictions pixel-wise (mean rule after sigmoid), prints per-dataset Dice and overall mean. The averaged predictions are saved in `<out_folder>/<dataset>/`, and `dice.csv` (ensemble name, models used, dataset, dice, plus the mean) in `<out_folder>/`. The Dice is computed image by image, so large datasets do not have to fit in memory.

### ⚙️ Sweep Configuration (`configs/sweep_train.yaml`, `configs/sweep_test.yaml`)

Experiments are driven by two files, one per stage:
- [`configs/sweep_train.yaml`](configs/sweep_train.yaml) — `models` + `training` axes (used by `run_training.py`)
- [`configs/sweep_test.yaml`](configs/sweep_test.yaml) — `models` + `testing` axes (used by `run_inference.py`)

The `models` list is duplicated across both files on purpose and must stay in sync
(same names, folders, scripts, config path).

Key fields:
- `models` — which models to run: their folder, scripts, per-model `config` yaml, and whether they use SAM augmentation (`has_aux: true/false`)
- `training.seeds` — one seed per run for reproducibility across the 5 runs
- `aux_sources` — aux models only: map from aux source (folder under `output/augmentation/`) to its own list of methods. A map and not `sources × methods` because SAM methods and depth methods only exist under their own source. Comment/uncomment sources or methods to select them, or use `--aux_source` on the CLI:
  ```yaml
  aux_sources:
    sam2:
      - "SAMAug"
      - "RG_logits"
    da3mono-large:
      - "DepthAug"
  ```
- `da_methods` / `lr_methods` — DA and LR axes; `sweep_test.da_methods` must match `sweep_train.da_methods` to find the checkpoints

If `has_aux: true` → runner loops over `(aux_source, method) pairs × da_methods × lr_methods × runs`.
If `has_aux: false` → runner loops over `da_methods × lr_methods × runs`.

New aux images (a new `aux_source`) need no code: generate them under `output/augmentation/<aux_source>/<method>/<dataset>/images/` (PNG, same file names as the dataset images, for the train set and every test set) and add the source to `aux_sources` in both sweep files.

Checkpoint naming (`build_model_name`), the aux axis (`get_aux_sources`) and combo validity (`is_valid_combo`) live in
[`src/sweep/naming.py`](src/sweep/naming.py), shared by both runners so the training
and testing stages can never disagree on filenames.

For SLURM: submit with `--array=1-5` and pass `$SLURM_ARRAY_TASK_ID` as `--run_id`. Each array job runs all method combinations for that run ID in parallel with the others.

## 🔧 Available Augmentation Methods

### SAM-based methods

Built from the SAM priors in [`src/augmentation/sam_prior_methods.py`](src/augmentation/sam_prior_methods.py) (the original versions, which run SAM inside every method, are still in [`sam_methods.py`](src/augmentation/sam_methods.py)):

- `SAMAug` - G += segPrior, B += boundaryPrior
- `ourSAMAug` - as SAMAug, plus R += areaPrior (relative area of the masks)
- `RG_segPrior` - original R and G, B = segPrior
- `SV_segPrior` -
- `RG_logits` - original R and G, B = SAM logits (min-max)
- `PCA_segPrior` - PCA 3→2 of the RGB, B = segPrior

SAM priors: `segPrior` = sum of the stability scores of the overlapping masks, `boundaryPrior` = contours of the masks {0,1}, `areaPrior` = relative area of the masks, `logits` = logits of the SAM predictor on the whole image.

### Depth-based methods

From the Depth Anything 3 priors, in [`src/augmentation/depth_prior_methods.py`](src/augmentation/depth_prior_methods.py):

- `DepthAug` - G += frst_prior, B += localnorm
- `RG_voteV3` - original R and G, B = frst_prior
- `RG_depth` - original R and G, B = depth_p1p99
- `RG_localnorm` - original R and G, B = localnorm
- `PCA_voteV3` - PCA 3→2 of the RGB, B = frst_prior
- `LAB_gamut_aplus` - in Lab, a* moved towards the reddest in-gamut value, proportionally to frst_prior

Depth priors: `frst_prior` = region prior (radial voting on the depth gradient + back-projection), `localnorm` = boundary prior (gradient / local Gaussian mean), `depth_p1p99` = reversed depth (near = 1), all in [0,1] per image.

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
- **HGM only** (separate conda env, [`requirements_hgm.txt`](requirements_hgm.txt)): torch 2.2.1 cu121, `mamba_ssm`, `torch_geometric`, `dynamic-network-architectures`

## 📝 Notes

- SAM checkpoints are automatically downloaded to `checkpoints_sam/`
- Dataset structure must follow the format: `{dataset}/images/` and `{dataset}/masks/`
- All generated outputs (models, predictions, augmented data) are saved in `output/` and are gitignored
- All 4 Test scripts save raw logits (pre-sigmoid, float32, `.npy`) in `output/logits/`
- PolypPVT scripts are complete but not yet validated on the cluster
- Runner scripts have not yet been validated on the cluster

---

*Last updated: 19 Ottobre 2026*
