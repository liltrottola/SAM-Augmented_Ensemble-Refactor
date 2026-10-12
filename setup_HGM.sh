#!/bin/bash
# Setup of the separate conda env for HGM (mamba_ssm, torch_geometric, dynamic-network-architectures).
# Launch from the repo root:  mkdir -p logs && sbatch slurm/run_setup_hgm.slurm

# Initialize conda (generic: works with any conda installation available in PATH)
eval "$(conda shell.bash hook)"

# NEW: create the env if missing (python 3.10, as in the original HGM README)
if ! conda env list | grep -q "^HGM "; then
    conda create -n HGM python=3.10 -y
fi
conda activate HGM

pip install numpy==2.2.2

echo "=== [1/4] Installing libtiff ==="
conda install -c conda-forge libtiff -y

echo "=== [2/4] Installing PyTorch cu121 ==="     # FIX: was "cu130", the index-url is cu121
pip install torch==2.2.1 torchvision==0.17.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== [3/4] Installing mamba_ssm ==="
pip install mamba_ssm==2.2.4 --no-build-isolation

echo "=== [4/4] Installing requirements_hgm.txt ==="
pip install -r requirements_hgm.txt

echo "==============================="
echo "=== FINAL CHECK ==="
python -c "
import torch
print('torch:          ', torch.__version__)
print('cuda available: ', torch.cuda.is_available())
print('cuda version:   ', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU:            ', torch.cuda.get_device_name(0))
"
echo "=== DONE - $(date) ==="
echo "==============================="
