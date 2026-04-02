#!/bin/bash
#SBATCH --job-name=HSNETAux_TRAIN_debug
#SBATCH --output=logs/hsnetAux_sam2_ourSAMAug_%j.out
#SBATCH --error=logs/hsnetAux_sam2_ourSAMAug%j.err
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

echo "Avvio training di PolypPVT Aug..."
mkdir -p logs
python src/models/PolypPVT/Train_aux.py

echo "training HSNET Aux COMPLETATO!"