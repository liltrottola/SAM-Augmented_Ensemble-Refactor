#!/bin/bash
#SBATCH --job-name=ORACLE
#SBATCH --output=logs/oracle_%j.out
#SBATCH --error=logs/oracle_%j.err
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

echo "Starting process: ${MODE}"

python scripts/run_oracle.py --config ../configs/ensemble.yaml --mode ${MODE:-oracle1}

echo "Process ${MODE} completed succesfully!"
