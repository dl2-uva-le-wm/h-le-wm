#!/bin/bash
# Super simple eval job for your Hi-LeWM L2 d25 checkpoint on PushT.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=hi_eval_l2_d25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=out/hi_eval_l2_d25_%j.out
#SBATCH --error=out/hi_eval_l2_d25_%j.err

set -eo pipefail

mkdir -p out

module purge
module load 2025
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
conda activate lewm-gpu

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"

# Assumes you submit from repo root:
#   sbatch jobs/eval/hi/pusht_eval_l2_d25_simple.sh
cd "${SLURM_SUBMIT_DIR}"

python hi_eval.py \
  --config-name=hi_pusht \
  policy=hi_lewm_l2_d25_epoch_1 \
  wm.num_levels=2
