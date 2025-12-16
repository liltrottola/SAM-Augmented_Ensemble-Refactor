# SAM-Augmented Ensemble (Refactoring)

⚠️ **PROJECT UNDER REFACTORING** ⚠️

## 📂 Original Project

https://github.com/LorisNanni/Exploring-SAM-Augmented-Ensembles

This repository contains a framework for augmenting medical image datasets using Segment Anything Model (SAM/SAM2) and training segmentation models (HSNet, PolypPVT).

**The project is currently being restructured** to improve modularity, readability, and code maintainability. The final structure and some components may be subject to changes.

## 📋 Project Structure

```
├── configs/                    # YAML configuration files
│   ├── augmentation.yaml      # Configuration for augmentation
│   └── hsnet_vanilla.yaml     # Configuration for HSNet training
├── datasets/                  # Original datasets
├── scripts/                   # Executable scripts
│   └── run_augmentation.py   # Main script for augmentation
├── src/                      # Modular source code
│   ├── augmentation/         # Augmentation modules
│   │   ├── methods.py       # Augmentation methods
│   │   ├── sam_loader.py    # SAM model loading
│   │   └── out/             # Augmentation output directory
│   └── models/              # Segmentation models
│       ├── HSNet/           # HSNet model
│       │   ├── Train.py     # Training script
│       │   ├── Test.py      # Testing script
│       │   ├── lib/         # Model libraries
│       │   ├── utils/       # Utilities
│       │   ├── model_pth/   # Model weights directory
│       │   └── pretrained_pth/  # Pretrained weights
│       ├── HSNet_aux/       # HSNet auxiliary models
│       ├── PolypPVT/        # PolypPVT model
│       └── SAM/             # SAM model
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

**Output:** Augmented datasets are saved in `src/augmentation/out/sam{1,2}/{method}/`

### Model Training

**Script to run:** [`src/models/HSNet/Train.py`](src/models/HSNet/Train.py)

**Configuration:** Edit the [`configs/hsnet_vanilla.yaml`](configs/hsnet_vanilla.yaml) file to specify training parameters.

**Execution:**

⚠️ **TEMPORARY NOTE:** For now, training must be started **from inside the HSNet model folder** due to relative paths that have not yet been updated. All relative directories will be modified later to allow launching directly from the project home.

```bash
cd src/models/HSNet
python Train.py --config ../../../configs/hsnet_vanilla.yaml
```

## 🔧 Available Augmentation Methods

Methods implemented in [`src/augmentation/methods.py`](src/augmentation/methods.py):

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

---

*Last updated: December 16, 2025*
