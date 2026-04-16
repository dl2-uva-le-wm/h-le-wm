#!/bin/bash

# Snellius job: Train Hi-LeWM on PushT with 2-level topology for d=25 env steps.
# d=25 with frameskip=5 -> k2=5 model steps.
#
# Usage:
#   cd jobs/2_levels/pusht
#   sbatch train_d25.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=hi_l2_pusht_d25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=out/train_d25_%j.out
#SBATCH --error=out/train_d25_%j.err

set -eo pipefail

mkdir -p out

module purge
module load 2025
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
conda activate lewm

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"

cd ../../..

echo "STABLEWM_HOME=${STABLEWM_HOME}"
echo "Launching: num_levels=2, k2=5 (d=25)"

python hi_train.py \
  wm.num_levels=2 \
  wm.k1=0 \
  wm.k2=5 \
  data=hi_pusht \
  output_model_name=hi_lewm_l2_d25
