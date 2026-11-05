#!/bin/bash
#SBATCH --job-name=DAv3_Setup
#SBATCH --output=logs/setup_dav3_%j.out
#SBATCH --error=logs/setup_dav3_%j.err
#SBATCH --partition=allgroups      # CPU only: CUDA is checked by the first depth job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                  # pip needs RAM (see run_setup.slurm)
#SBATCH --time=02:00:00

# Creates venv_dav3 and checkpoints_dav3/ in the repo root (setup_DAv3.sh).
# Launch from the repo root:  mkdir -p logs && sbatch slurm/run_setup_dav3.slurm

cd $SLURM_SUBMIT_DIR

echo "Starting DA3 env setup on node: $HOSTNAME"
bash setup_DAv3.sh

echo "DA3 env setup completed"
