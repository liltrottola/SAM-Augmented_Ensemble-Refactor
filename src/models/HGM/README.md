# HGM (Hybrid Graph Mamba)

Polyp segmentation model from *"Hybrid Graph Mamba: Unlocking Non-Euclidean Potential for
Accurate Polyp Segmentation"* (MICCAI 2025). PVTv2-b2 backbone + Mamba/GCN modules, single output map.

## Environment
HGM needs its own conda env (`mamba_ssm`, `torch_geometric`, `dynamic-network-architectures`,
torch 2.2.1 cu121): it does NOT run in `venv_newSAMAug`.
- setup: `mkdir -p logs && sbatch slurm/run_setup_hgm.slurm` (runs `setup_HGM.sh`, requirements in `requirements_hgm.txt`).
  The script creates a conda env named `HGM` in the conda installation available in PATH.
- activate: `conda activate HGM`

## Pretrained weights
`pretrained_pth/pvt_v2_b2.pth`: ImageNet PVTv2-b2 backbone, same file used by HSNet/PolypPVT.
The vanilla training starts from it only (the authors' `best_parameter_HGM.pth` is NOT loaded).

## Usage
Config: `configs/hgm_vanilla.yaml`. Always launch with `--model hgm` from the HGM env
(sweep entry `hgm` in `configs/sweep_train.yaml` / `sweep_test.yaml`).
- train: `python Train_vanilla.py [--debug] [--model_name NAME] [--online_da da3|noda] [--offline_da da1|da2|clahe|noda] [--lr_method ...]`
- test:  `python Test_vanilla.py --model_pth NAME.pth`

## Changes w.r.t. the original code
- Train/Test rewritten on the HSNet `Train_vanilla.py`/`Test_vanilla.py` structure (yaml config, seed, offline/online DA, LR methods).
- Dataloader: HSNet `utils/dataloader.py` (with `noda` identical to the original HGM `PolypDataset`; adds `da3`).
- `lib/pvt.py`: backbone weights path `./data/` -> `./pretrained_pth/`.
- No DataParallel (single GPU), no `--pre_parameter`.
- LR: original `adjust_lr` (decay_epoch 50) replaced by HSNet-style milestones `[50, 100]`, decay 0.1: same LR per epoch.
- Optimizer `Adam` lr 3e-5, no weight decay, `cudnn.enabled = False`, as original.
- `num_workers: 4` as original (HSNet/PolypPVT use 1).
- Not ported: original `validation()`/`EvalDataset` (binary metrics), ASPS/storage/patch dataloaders.
