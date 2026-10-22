#!/bin/bash
set -euo pipefail

echo "CREATING VIRTUAL ENVIRONMENT (VENV) for DSINE"
if [ ! -d "venv_dsine" ]; then
    python3.11 -m venv venv_dsine
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi
source venv_dsine/bin/activate

if [[ "$(which python)" != *"venv_dsine"* ]]; then
    echo "ERRORE: l'ambiente attivo non e' venv_dsine."
    echo "python attuale: $(which python)"
    exit 1
fi

echo "CLONING DSINE REPO"
if [ ! -d "DSINE" ]; then
    git clone https://github.com/baegwangbin/DSINE.git
    echo "DSINE repo cloned."
else
    echo "DSINE repo already exists."
fi

echo "PATCHING DSINE PATHS"
DSINE_ABS="$(cd DSINE && pwd)"
PATCH_FILE=$(grep -rl "EXPERIMENT_DIR = '/home/gwangbin" DSINE --include="*.py" | head -n 1 || true)
if [ -n "$PATCH_FILE" ]; then
    sed -i "s|^DATASET_DIR = .*|DATASET_DIR = '${DSINE_ABS}/datasets/'|; s|^EXPERIMENT_DIR = .*|EXPERIMENT_DIR = '${DSINE_ABS}/experiments_out/'|" "$PATCH_FILE"
    echo "DSINE paths patched in $PATCH_FILE"
else
    echo "DSINE paths already patched."
fi

echo "INSTALLING DSINE REQUIREMENTS"
python -m pip install torch torchvision torchaudio \
	  --index-url https://download.pytorch.org/whl/cu121 \
	 --extra-index-url https://pypi.org/simple

pip install geffnet gdown pyyaml

echo "GETTING DSINE CHECKPOINTS"
CKPT_FOLDER_ID="1Gi9oYit86F9_fZtcCvKyxfy_r_YI0oxV"
CKPT_DIR="DSINE/projects/dsine/checkpoints"
if [ ! -f "$CKPT_DIR/exp001_cvpr2024/dsine.pt" ]; then
    mkdir -p "$CKPT_DIR"
    gdown --folder "https://drive.google.com/drive/folders/${CKPT_FOLDER_ID}" -O "$CKPT_DIR"
    echo "DSINE checkpoints downloaded."
else
    echo "DSINE checkpoints already exist."
fi

echo "The DSINE environment must be activated before running the DSINE scripts only."
echo "To activate it yourself, run: source venv_dsine/bin/activate"
