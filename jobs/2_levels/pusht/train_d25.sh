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
conda activate lewm-gpu

####################################### WANDB SETUP #######################################
WANDB_ENV_FILE="${WANDB_ENV_FILE:-$HOME/.config/wandb.env}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  set -a
  source "${WANDB_ENV_FILE}"
  set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set." >&2
  echo "Set it in ${WANDB_ENV_FILE} or submit with: sbatch --export=ALL,WANDB_API_KEY=<your_key> train_d25.sh" >&2
  exit 2
fi
wandb login --relogin "${WANDB_API_KEY}"

# Optional: set WANDB_ENTITY only if you want to force a specific workspace.
# By default we pass Hydra null so W&B uses the logged-in default entity.
WANDB_ENTITY_OVERRIDE="${WANDB_ENTITY:-null}"
WANDB_PROJECT="${WANDB_PROJECT:-hi_lewm}"

######################################## ENV SETUP & TRAINING LAUNCH #######################################

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"

cd ../..

echo "STABLEWM_HOME=${STABLEWM_HOME}"
echo "Launching: num_levels=2, k2=5 (d=25)"
echo "W&B entity: ${WANDB_ENTITY:-<default from login>}"
echo "W&B project: ${WANDB_PROJECT}"

python ../hi_train.py \
  wm.num_levels=2 \
  wm.k1=0 \
  wm.k2=5 \
  data=hi_pusht \
  wandb.config.entity=${WANDB_ENTITY_OVERRIDE} \
  wandb.config.project=${WANDB_PROJECT} \
  output_model_name=hi_lewm_l2_d25
