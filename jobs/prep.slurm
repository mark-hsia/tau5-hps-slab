#!/bin/bash
#SBATCH --job-name=tau5_prep
#SBATCH --account=robustelli
#SBATCH --partition=a5000
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err
#SBATCH --mail-type=END,FAIL

# Benchmarked ~2 h total; 12 h requested for headroom.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate hps

echo "host      : $(hostname)"
echo "gpu       : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "openmm    : $(python -c 'import openmm;print(openmm.version.version)')"
echo "started   : $(date)"

python scripts/01_prep.py --max-hours 11.0

echo "finished  : $(date)"
