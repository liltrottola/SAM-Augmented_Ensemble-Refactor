#!/bin/bash
#SBATCH --job-name=HSNET_TRAIN
#SBATCH --output=logs/hsnet_train_%A_%a.out    # %A = job ID, %a = array index
#SBATCH --error=logs/hsnet_train_%A_%a.err
#SBATCH --array=1-5
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00

cd $SLURM_SUBMIT_DIR
mkdir -p logs

source venv_newSAMAug/bin/activate

echo "Run $SLURM_ARRAY_TASK_ID - avvio training HSNet vanilla..."
python scripts/run_training.py \
    --model hsnet \
    --run_id $SLURM_ARRAY_TASK_ID

echo "Run $SLURM_ARRAY_TASK_ID - COMPLETATO"
