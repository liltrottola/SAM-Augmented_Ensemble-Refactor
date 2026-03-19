#!/bin/bash
#SBATCH --job-name=HSNET_TRAIN
#SBATCH --output=logs/hsnet_%j.out
#SBATCH --error=logs/hsnet_%j.err
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00

cd $SLURM_SUBMIT_DIR

echo "Attivo VENV..."
source venv_newSAMAug/bin/activate

echo "Avvio training di HSNet base..."
mkdir -p logs
python src/models/HSNet/Train.py

echo "training HSNET COMPLETATO!"