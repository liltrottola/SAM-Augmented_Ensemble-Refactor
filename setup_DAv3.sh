#!/bin/bash
set -euo pipefail

echo "CREATING VIRTUAL ENVIRONMENT (VENV) for Depth Anything model"
if [ ! -d "venv_dav3" ]; then
    python3.11 -m venv venv_dav3
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo "ACTIVATING VENV for DEPTH ANYTHING MODEL NOW"
source venv_dav3/bin/activate

if [[ "$(which python)" != *"venv_dav3"* ]]; then
    echo "ERRORE: l'ambiente attivo non e' venv_dav3."
    echo "python attuale: $(which python)"
    exit 1
fi

echo "CLONING DAv3 REPO"
if [ ! -d "depth-anything-3" ]; then
    git clone https://github.com/ByteDance-Seed/depth-anything-3.git
    echo "DAv3 repo cloned."
else
    echo "DAv3 repo already exists."
fi

echo "INSTALLING DAv3 REQUIREMENTS"
cd depth-anything-3
pip install -e .
cd ..

echo "GETTING DAv3 CHECKPOINTS"
mkdir -p checkpoints_dav3

if [ ! -f "checkpoints_dav3/da3mono-large/model.safetensors" ]; then
    hf download depth-anything/da3mono-large --local-dir checkpoints_dav3/da3mono-large
    echo "DAv3 checkpoint downloaded."
else
    echo "DAv3 checkpoint already exists."
fi

echo "The DAv3 environment must be activated before running the DAv3 model scripts only."
echo "To activate it yourself, run: source venv_dav3/bin/activate"