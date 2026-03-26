#!/bin/bash
#SBATCH --job-name=HSNET_TEST
#SBATCH --output=logs/hsnet_test_%A_%a.out
#SBATCH --error=logs/hsnet_test_%A_%a.err
#SBATCH --array=1-5
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0:30:00

cd $SLURM_SUBMIT_DIR

source venv_newSAMAug/bin/activate

echo "Run $SLURM_ARRAY_TASK_ID - avvio inference HSNet..."
python scripts/run_inference.py \
    --model hsnet \
    --run_id $SLURM_ARRAY_TASK_ID

echo "Run $SLURM_ARRAY_TASK_ID - COMPLETATO"
