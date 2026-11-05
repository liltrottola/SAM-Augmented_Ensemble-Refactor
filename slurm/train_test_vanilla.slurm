#!/bin/bash
#SBATCH --job-name=train_test_vanilla
#SBATCH --output=logs/train_test_vanilla_%j.out
#SBATCH --error=logs/train_test_vanilla_%j.err
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6          # 4 dataloader workers + main process
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# Example: train + test of ONE vanilla model (RGB only), ONE run.
# Copy this file and change only the block below; every axis is fixed here, nothing is taken from the sweep yamls.
# Launch from the repo root:  mkdir -p logs && sbatch slurm/train_test_vanilla.slurm

# ---------------- what to run ----------------
MODEL="hsnet"                # hsnet | polypvt | hgm | sam2unet | sam2rfanet (names in configs/sweep_registry.yaml)
RUN=1                        # run_id, the seed is seeds[RUN-1] of configs/sweep_train.yaml
TRAIN_SET="TrainDataset"     # folder under datasets/
DA="da3"                     # noda | da3 (online) | da1 | da2 | clahe | light | gpt (offline, datasets/<DA>/<TRAIN_SET>)
LR="lrbase"                  # lrbase | lra | lrb | lrc
# ---------------------------------------------

ARGS="--train_dataset $TRAIN_SET --da_method $DA --lr_method $LR"

cd $SLURM_SUBMIT_DIR
mkdir -p logs

# HGM: replace with its conda env
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source venv_newSAMAug/bin/activate

echo "Train $MODEL run $RUN ($ARGS)"
python -u scripts/run_training.py --model $MODEL --run_id $RUN $ARGS || exit 1

echo "Test $MODEL run $RUN ($ARGS)"
python -u scripts/run_inference.py --model $MODEL --run_id $RUN $ARGS

echo "Done"
