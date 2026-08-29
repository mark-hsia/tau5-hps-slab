#!/bin/bash
#SBATCH --job-name=tau5_s310
#SBATCH --account=robustelli
#SBATCH --partition=2080ti
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slab_one_%j.out
#SBATCH --error=logs/slab_one_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue

# Single 310 K run on turing01 (2080ti, non-preemptable, 7d limit).
# Measured 7.3 us/day on the 2080 Ti -> ~33 h for 10 us.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

T=${SLAB_T:?set SLAB_T}
VARIANT=${SLAB_VARIANT:-}

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate hps

echo "host    : $(hostname)"
echo "gpu     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "T       : $T K"
echo "variant : ${VARIANT:-WT}"
echo "started : $(date)"

python scripts/02_slab.py --temperature "$T" --steps 1000000000 --max-hours 46.0 ${VARIANT:+--variant "$VARIANT"}

echo "finished: $(date)"
