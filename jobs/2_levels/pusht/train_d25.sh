#!/bin/bash

# Snellius job: Train Hi-LeWM on PushT with 2-level topology for d=25 env steps.
# d=25 with frameskip=5 -> k2=5 model steps.
#
# Usage:
#   sbatch jobs/2_levels/pusht/train_d25.sh
# Recommended:
#   sbatch --export=ALL,PROJECT_ROOT=$PWD jobs/2_levels/pusht/train_d25.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=hi_l2_pusht_d25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=jobs/2_levels/pusht/out/train_d25_%j.out
#SBATCH --error=jobs/2_levels/pusht/out/train_d25_%j.err

set -eo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
conda activate lewm

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"

REPO_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
if [[ ! -f "hi_train.py" ]]; then
  echo "ERROR: hi_train.py not found in ${REPO_ROOT}" >&2
  echo "Submit from repo root or pass PROJECT_ROOT=/path/to/h-lewm" >&2
  exit 2
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "STABLEWM_HOME=${STABLEWM_HOME}"
echo "Launching: num_levels=2, k2=5 (d=25)"

python hi_train.py \
  wm.num_levels=2 \
  wm.k1=0 \
  wm.k2=5 \
  data=hi_pusht \
  output_model_name=hi_lewm_l2_d25
