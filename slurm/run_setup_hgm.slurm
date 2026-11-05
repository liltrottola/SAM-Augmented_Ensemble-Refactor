#!/bin/bash
#SBATCH --job-name=HGM_Setup
#SBATCH --output=logs/setup_hgm_%j.out
#SBATCH --error=logs/setup_hgm_%j.err
#SBATCH --partition=allgroups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx             # mamba_ssm may build CUDA kernels if no wheel matches
#SBATCH --cpus-per-task=4          # 4 CPUs to compile faster
#SBATCH --mem=32G                  # 32GB RAM (CRUCIAL to avoid pip crashing)
#SBATCH --time=01:00:00

# Launch from the repo root:  mkdir -p logs && sbatch slurm/run_setup_hgm.slurm

cd $SLURM_SUBMIT_DIR

echo "Starting HGM env setup on node: $HOSTNAME"
bash setup_HGM.sh

echo "HGM env setup completed"
