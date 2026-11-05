#!/bin/bash
#SBATCH --job-name=train_test_aux
#SBATCH --output=logs/train_test_aux_%j.out
#SBATCH --error=logs/train_test_aux_%j.err
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6          # 4 dataloader workers + main process
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# Example: train + test of ONE aux model (RGB + aux image), ONE run.
# Copy this file and change only the block below; every axis is fixed here, nothing is taken from the sweep yamls.
# Without --aux_source / --aux_methods the runners would launch ALL aux sources with all their methods.
# The aux images must exist for the train set and for every test set:
#   output/augmentation/<AUX_SOURCE>/<AUX_METHOD>/<dataset>/
# Launch from the repo root: sbatch slurm/train_test_aux.slurm

# ---------------- what to run ----------------
MODEL="hsnet_aux"            # hsnet_aux | polypvt_aux | hgm_aux | sam2unet_aux | sam2rfanet_aux (names in configs/sweep_registry.yaml)
RUN=1                        # run_id, the seed is seeds[RUN-1] of configs/sweep_train.yaml
TRAIN_SET="TrainDataset"     # folder under datasets/
DA="noda"                     # noda | da1 | da2 | clahe | light | gpt (offline, datasets/<DA>/<TRAIN_SET>)
LR="lrbase"                  # lrbase | lra | lrb | lrc
AUX_SOURCE="sam2_stab085"    # folder under output/augmentation/ (aux_sources of configs/sweep_registry.yaml)
AUX_METHOD="SAMAug"          # one method of AUX_SOURCE
# ---------------------------------------------

ARGS="--train_dataset $TRAIN_SET --da_method $DA --lr_method $LR --aux_source $AUX_SOURCE --aux_methods $AUX_METHOD"

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
