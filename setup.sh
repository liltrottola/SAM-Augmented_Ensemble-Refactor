#!/bin/bash

echo "CREATING VIRTUAL ENVIRONMENT (VENV)"
if [ ! -d "venv_newSAMAug" ]; then
    python3.11 -m venv venv_newSAMAug
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo "ACTIVATING VENV NOW"
source venv_newSAMAug/bin/activate




echo "CLONING SAM2 REPO"
if [ ! -d "segment-anything-2" ]; then
    git clone https://github.com/facebookresearch/segment-anything-2.git
    echo "SAM2 repo cloned."

    echo "INSTALLING SAM2 REQUIREMENTS"
    cd segment-anything-2
    pip install -e .
    cd ..
else
    echo "SAM2 repo already exists."
fi

echo "INSTALLING SAMAug REQUIREMENTS"
pip install -r requirements.txt
pip install segment_anything
pip install scikit-image
pip install matplotlib
pip install scikit-learn
 
echo "GETTING SAM2 CHECKPOINT"
mkdir -p checkpoints_sam
cd checkpoints_sam
if [ ! -f "sam2_hiera_large.pt" ]; then
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    echo "SAM2 checkpoint downloaded."
else
    echo "SAM2 checkpoint already exists."
fi

echo "GETTING SAM1 CHECKPOINT"
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    echo "SAM1 checkpoint downloaded."
else
    echo "SAM1 checkpoint already exists."
fi
cd ..


echo "The VENV must be activated before running all other scripts."
echo "To activate it yourself, run: source venv_newSAMAug/bin/activate"