#!/bin/bash
#SBATCH --job-name=SAM_Setup       # Job name in the queue
#SBATCH --output=logs/setup_%j.out # Where to save the output (creates the logs folder)
#SBATCH --error=logs/setup_%j.err  # Where to save the errors
#SBATCH --partition=allgroups      # Standard compute partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # 4 CPUs to compile faster
#SBATCH --mem=32G                  # 32GB of RAM (CRUCIAL to avoid pip crashing)
#SBATCH --time=01:00:00            # Maximum run time


# 2. Move to the folder the job was submitted from
cd $SLURM_SUBMIT_DIR

# 3. Make sure setup.sh is executable
chmod +x setup.sh

echo "Starting installation on node: $HOSTNAME"
echo "Now running setup.sh inside Slurm with 32GB of RAM..."
./setup.sh

echo "Installation completed"
