#!/bin/bash
#SBATCH --job-name=ENSEMBLE
#SBATCH --output=logs/ensemble_%j.out
#SBATCH --error=logs/ensemble_%j.err
#SBATCH --partition=allgroups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

cd $SLURM_SUBMIT_DIR

echo "Node: $HOSTNAME"
echo "Activating VENV..."
source venv_newSAMAug/bin/activate

echo "Starting ENSEMBLE process"

python scripts/run_ensemble.py --config ../configs/ensemble.yaml

echo "Ensemble completed succesfully!"
