#!/bin/bash
#SBATCH --job-name=tau5_slab
#SBATCH --account=robustelli
#SBATCH --partition=a5000
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/slab_task%a_%A.out
#SBATCH --error=logs/slab_task%a_%A.err
#SBATCH --mail-type=END,FAIL

# %2 caps concurrency at two: amp02 idle, amp03 mixed, amp01 draining.
# Runs are independent, so serialising the third costs nothing.
# Measured 7.2 us/day in the 10x10x50 nm slab box -> ~33 h for 10 us.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TEMPS=(280 310 350)
T=${TEMPS[$SLURM_ARRAY_TASK_ID]}

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate hps

echo "host    : $(hostname)"
echo "gpu     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "T       : $T K"
echo "started : $(date)"

python scripts/02_slab.py --temperature "$T" --steps 1000000000 --max-hours 46.0

echo "finished: $(date)"
