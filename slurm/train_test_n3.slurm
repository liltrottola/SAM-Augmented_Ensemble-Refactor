#!/bin/bash
#SBATCH --job-name=train_test_n3
#SBATCH --output=logs/train_test_n3_%j.out
#SBATCH --error=logs/train_test_n3_%j.err
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6          # 4 dataloader workers + main process
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# Example: train + test of ONE N3 model (RGB + prior channels -> ChannelReduction -> model), ONE run.
# Copy this file and change only the block below; every axis is fixed here, nothing is taken from the sweep yamls.
# Prior names per family as in prior_sources of configs/sweep_registry.yaml (the family folder is resolved there);
# leave a family empty to not use it. Block settings (residual, kernel, LR mult) are in configs/<model>_n3.yaml.
# The priors must exist for the train set and for every test set: output/priors/<family folder>/<dataset>/<prior>/
# Launch from the repo root: sbatch slurm/train_test_n3.slurm

# ---------------- what to run ----------------
MODEL="hsnet_n3"                     # hsnet_n3 | polypvt_n3 | hgm_n3 | sam2unet_n3 | sam2rfanet_n3 (names in configs/sweep_registry.yaml)
RUN=1                                # run_id, the seed is seeds[RUN-1] of configs/sweep_train.yaml
TRAIN_SET="TrainDataset"             # folder under datasets/
DA="da3"                             # noda | da3 (online) | da1 | da2 | clahe | light | gpt (offline, datasets/<DA>/<TRAIN_SET>)
LR="lrbase"                          # lrbase | lra | lrb | lrc
SAM_PRIORS="segPrior boundaryPrior"  # segPrior boundaryPrior areaPrior logits
DEPTH_PRIORS="frst_prior localnorm"  # frst_prior localnorm depth_p1p99 shapeindex meancurvature meancurvature_p5p95
DSINE_PRIORS=""                      # normal_nz normal_mean7
# ---------------------------------------------

ARGS="--train_dataset $TRAIN_SET --da_method $DA --lr_method $LR"
if [ -n "$SAM_PRIORS" ];   then ARGS="$ARGS --sam_priors $SAM_PRIORS"; fi
if [ -n "$DEPTH_PRIORS" ]; then ARGS="$ARGS --depth_priors $DEPTH_PRIORS"; fi
if [ -n "$DSINE_PRIORS" ]; then ARGS="$ARGS --dsine_priors $DSINE_PRIORS"; fi

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
