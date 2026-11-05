#!/bin/bash
#SBATCH --job-name=DSINE_Setup
#SBATCH --output=logs/setup_dsine_%j.out
#SBATCH --error=logs/setup_dsine_%j.err
#SBATCH --partition=allgroups      # CPU only: CUDA is checked by the first normal generation job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                  # pip needs RAM (see run_setup.slurm)
#SBATCH --time=02:00:00

# Creates venv_dsine and DSINE/ (repo + checkpoints) in the repo root (setup_DSINE.sh).
# Launch from the repo root:  mkdir -p logs && sbatch slurm/run_setup_dsine.slurm

cd $SLURM_SUBMIT_DIR

echo "Starting DSINE env setup on node: $HOSTNAME"
bash setup_DSINE.sh

echo "DSINE env setup completed"
