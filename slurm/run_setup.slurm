#!/bin/bash
#SBATCH --job-name=SAM_Setup       # Nome in coda
#SBATCH --output=logs/setup_%j.out # Dove salva l'output (crea la cartella logs)
#SBATCH --error=logs/setup_%j.err  # Dove salva gli errori
#SBATCH --partition=allgroups      # Partizione di calcolo standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # 4 CPU per compilare più in fretta
#SBATCH --mem=32G                  # 32GB di RAM (CRUCIALE per evitare il crash di pip)
#SBATCH --time=01:00:00            # Tempo massimo


# 2. Spostati nella cartella da cui hai lanciato il comando
cd $SLURM_SUBMIT_DIR

# 3. Assicurati che setup.sh sia eseguibile
chmod +x setup.sh

echo "Inizio installazione su nodo: $HOSTNAME"
echo "Ora parte setup.sh dentro Slurm con 32GB di RAM..."
./setup.sh

echo "Installazione completata"