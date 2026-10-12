# SAM2-UNet

Segmentation model from *"SAM2-UNet: Segment Anything 2 Makes Strong Encoder for Natural and
Medical Image Segmentation"* (2024). Frozen SAM2 Hiera-L encoder with trainable adapters + U-Net decoder,
3 outputs (final head + 2 side outputs for deep supervision).

## Environment
Runs in the main `venv_newSAMAug` (same env as HSNet/PolypPVT): `source venv_newSAMAug/bin/activate`.

The bundled `sam2/` and `sam2_configs/` (old SAM2 copy shipped with SAM2-UNet) must stay in this folder:
the scripts run from here, so `import sam2` resolves to this local copy and not to the
`segment-anything-2` package installed in the venv.

## Pretrained weights
`paths.hiera_path` in the yaml -> `checkpoints_sam/sam2_hiera_large.pt` (downloaded by `setup.sh`).
Used only at training. At test the model is built without it: all weights (frozen encoder included)
are loaded from the trained `.pth`, as in the original `test.py`.

## Usage
Config: `configs/sam2unet_vanilla.yaml` (sweep entry `sam2unet` in `configs/sweep_train.yaml` / `sweep_test.yaml`).
- train: `python Train_vanilla.py [--debug] [--model_name NAME] [--online_da da3|noda] [--offline_da da1|da2|clahe|noda] [--lr_method ...]`
- test:  `python Test_vanilla.py --model_pth NAME.pth`

## Changes w.r.t. the original code
- Train/Test rewritten on the HSNet `Train_vanilla.py`/`Test_vanilla.py` structure (yaml config, seed, offline/online DA, LR methods).
- Same as original: AdamW lr 1e-3 / wd 5e-4, `CosineAnnealingLR(T_max=epochs, eta_min=1e-7)` stepped once per epoch
  (legacy path `lr_method: lrbase`), 20 epochs, batch 12, loss = sum of the 3 `structure_loss`, no gradient clipping.
- Multi-scale from yaml `size_rates`: `[0.75, 1.0, 1.25]` = original `train_rate.py`, `[1.0]` = original `train.py`.
- Dataloader: HSNet `utils/dataloader.py` instead of the original `FullDataset`:
  - `noda` = no augmentation at all (the original random h/v flips are NOT applied); `da3` available;
  - GT resized with bilinear interpolation (original: bicubic); `filter_files` drops image/GT size mismatches.
- Test: only the first output (final head) under `torch.no_grad()`, as original; probability maps
  (sigmoid, no min-max normalization) + logits `.npy`, same format as HSNet for `run_ensemble.py`.