#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:rtx:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00          # estimate (not measured): SAM2 ~2-3 s/img on all the sets dominates

# Data pipeline: offline DA, depth / normals / SAM priors, aux images. Every step reads its own config, the
# dataset lists are in datasets.folders of each yaml (nothing is set here):
#    1. DA1                 configs/da1_augmentation.yaml     -> datasets/da1/<dataset>/
#    2. DA2                 configs/da2_augmentation.yaml     -> datasets/da2/<dataset>/
#    3. DA3 depth (GPU)     configs/depth_generation.yaml     -> output/depth/da3mono-large/<dataset>/npy/
#    4. DSINE normals (GPU) configs/normal_generation.yaml    -> output/normals/dsine/<dataset>/npy/
#    5. depth priors        configs/depth_prior_maps.yaml     -> output/priors/da3mono-large/<dataset>/<prior>/
#    6. normal priors       configs/normal_prior_maps.yaml    -> output/priors/dsine/<dataset>/<prior>/
#    7. light DA            configs/light_augmentation.yaml   -> datasets/light/<dataset>/  (needs 3 and 4)
#    8. SAM priors (GPU)    configs/sam_generation.yaml       -> output/priors/sam2_stab085/<dataset>/<prior>/
#    9. SAM aux             configs/sam_augmentation.yaml     -> output/augmentation/sam2_stab085/<method>/<dataset>/
#   10. depth aux           configs/depth_augmentation.yaml   -> output/augmentation/da3mono-large/<method>/<dataset>/
#   11. normal aux          configs/normal_augmentation.yaml  -> output/augmentation/dsine/<method>/<dataset>/
#
# Light DA runs before SAM so that SAM is a single step on all the sets.
# gpt/<dataset> is not generated here: it must already be in datasets/.
# The scripts have no resume: if the job dies, relaunch from the failed step. Stops at the first failing step.
#
# Launch from the repo root:
#   sbatch slurm/pipeline.slurm          # all the steps
#   sbatch slurm/pipeline.slurm 5        # from step 5 to the end
#   sbatch slurm/pipeline.slurm 8 8      # only step 8 (SAM priors)
#   sbatch slurm/pipeline.slurm 9 11     # only the aux images

set -e

START_STEP=${1:-1}
END_STEP=${2:-11}

cd $SLURM_SUBMIT_DIR
mkdir -p logs


step() { echo; echo "=============== $1 ($(date +%H:%M:%S)) ==============="; }
run_step() { [ "$1" -ge "$START_STEP" ] && [ "$1" -le "$END_STEP" ]; }   # run_step N -> true if step N must run

# venvs and checkpoints expected in the repo root
for d in venv_newSAMAug venv_dav3 venv_dsine checkpoints_sam checkpoints_dav3 DSINE; do
    if [ ! -d "$d" ]; then echo "ERROR: $d/ not found in the repo root"; exit 1; fi
done

# -------- main venv: offline DA1 / DA2 --------
source venv_newSAMAug/bin/activate

if run_step 1; then
    step "1. DA1"
    python -u scripts/run_da1_augmentation.py --config configs/da1_augmentation.yaml
fi
if run_step 2; then
    step "2. DA2"
    python -u scripts/run_da2_augmentation.py --config configs/da2_augmentation.yaml
fi

deactivate

# -------- DA3 venv: depth --------
if run_step 3; then
    source venv_dav3/bin/activate
    step "3. DA3 depth"
    python -u scripts/run_depth_generation.py --config configs/depth_generation.yaml
    deactivate
fi

# -------- DSINE venv: normals --------
if run_step 4; then
    source venv_dsine/bin/activate
    step "4. DSINE normals"
    python -u scripts/run_normal_generation.py --config configs/normal_generation.yaml
    deactivate
fi

# -------- main venv: priors, light DA, SAM, aux images --------
source venv_newSAMAug/bin/activate

if run_step 5; then
    step "5. depth priors"
    python -u scripts/run_depth_prior_maps.py --config configs/depth_prior_maps.yaml
fi
if run_step 6; then
    step "6. normal priors"
    python -u scripts/run_normal_prior_maps.py --config configs/normal_prior_maps.yaml
fi
if run_step 7; then
    step "7. light DA"
    python -u scripts/run_light_augmentation.py --config configs/light_augmentation.yaml
fi
if run_step 8; then
    step "8. SAM priors"
    python -u scripts/run_sam_generation.py --config configs/sam_generation.yaml
fi
if run_step 9; then
    step "9. SAM aux"
    python -u scripts/run_sam_augmentation.py --config configs/sam_augmentation.yaml
fi
if run_step 10; then
    step "10. depth aux"
    python -u scripts/run_depth_augmentation.py --config configs/depth_augmentation.yaml
fi
if run_step 11; then
    step "11. normal aux"
    python -u scripts/run_normal_augmentation.py --config configs/normal_augmentation.yaml
fi

echo "Pipeline completed (steps $START_STEP -> $END_STEP)"
