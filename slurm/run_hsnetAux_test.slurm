#!/bin/bash
#SBATCH --job-name=HSNETAUX_TEST
#SBATCH --output=logs/test_hsnetAux_%j.out
#SBATCH --error=logs/test_hsnetAux_%j.err
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

echo "Node: $HOSTNAME"
echo "Attivo VENV..."
source venv_newSAMAug/bin/activate

echo "Starting HSNet inference..."
mkdir -p logs

python src/models/HSNet_aux/Test.py

echo "Inference of HSNetAux ----> DONE"