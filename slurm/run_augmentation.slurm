#!/bin/bash
#SBATCH --job-name=AUG
#SBATCH --output=logs/aug_ourSAMAug_%j.out
#SBATCH --error=logs/aug_ourSAMAug_%j.err
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00

cd $SLURM_SUBMIT_DIR

echo "Attivo VENV..."
source venv_newSAMAug/bin/activate

echo "COmincio augmentation"
mkdir -p logs
python scripts/run_augmentation.py

echo "Augmentation completata"