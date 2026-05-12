#!/bin/bash

# Snellius job: Train Hi-LeWM on Reacher with 2-level topology for d=25 env steps.
# d=25 with frameskip=5 -> k2=5 model steps.
#
# Usage:
#   cd jobs/2_levels/reacher
#   sbatch train_d25.sh
#
# Optional overrides:
#   sbatch --export=ALL,STABLEWM_HOME=/scratch-shared/$USER/stablewm_data train_d25.sh
#   sbatch --export=ALL,RUN_NAME=hi_lewm_reacher_l2_d25_custom train_d25.sh

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=hi_l2_reacher_d25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=out/train_d25_%j.out
#SBATCH --error=out/train_d25_%j.err

set -eo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${SUBMIT_DIR}/out"

resolve_repo_root() {
  local candidate
  for candidate in \
    "${PROJECT_ROOT:-}" \
    "${SUBMIT_DIR}" \
    "${SUBMIT_DIR}/../../.." \
    "${HOME}/h-lewm" \
    "/gpfs/home2/${USER}/h-lewm"; do
    [[ -n "${candidate}" ]] || continue
    candidate="$(cd -- "${candidate}" >/dev/null 2>&1 && pwd || true)"
    if [[ -n "${candidate}" && -f "${candidate}/hi_train.py" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root with hi_train.py" >&2
  echo "Checked PROJECT_ROOT='${PROJECT_ROOT:-}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', PWD='${PWD}'" >&2
  exit 2
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
if conda env list | grep -E '(^|[[:space:]])lewm-gpu([[:space:]]|$)' >/dev/null 2>&1; then
  conda activate lewm-gpu
elif conda env list | grep -E '(^|[[:space:]])lewm([[:space:]]|$)' >/dev/null 2>&1; then
  conda activate lewm
else
  echo "ERROR: Could not find conda environment 'lewm-gpu' or 'lewm'" >&2
  echo "Run jobs/setup/setup_env.sh first, or create the environment from environment-gpu.yml" >&2
  exit 2
fi

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

WANDB_ENTITY_OVERRIDE="${WANDB_ENTITY:-null}"
WANDB_PROJECT="${WANDB_PROJECT:-hi_lewm}"

######################################## DATA STAGING #######################################

SHARED_STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
SHARED_DATASET_PATH="${SHARED_STABLEWM_HOME}/reacher.h5"
SHARED_CKPT_PATH="${SHARED_STABLEWM_HOME}/reacher/lewm_object.ckpt"

if [[ ! -f "${SHARED_DATASET_PATH}" ]]; then
  echo "ERROR: missing dataset ${SHARED_DATASET_PATH}" >&2
  echo "Run setup first, for example:" >&2
  echo "  sbatch --export=ALL,STABLEWM_HOME=${SHARED_STABLEWM_HOME} ${REPO_ROOT}/jobs/setup/download_reacher.sh" >&2
  exit 3
fi

if [[ ! -f "${SHARED_CKPT_PATH}" ]]; then
  echo "ERROR: missing checkpoint ${SHARED_CKPT_PATH}" >&2
  echo "Run setup first, for example:" >&2
  echo "  sbatch --export=ALL,STABLEWM_HOME=${SHARED_STABLEWM_HOME} ${REPO_ROOT}/jobs/setup/download_reacher_model.sh" >&2
  exit 4
fi

LOCAL_STABLEWM_HOME="${TMPDIR:-/tmp}/${USER}_stablewm_data_${SLURM_JOB_ID:-manual}"
mkdir -p "${LOCAL_STABLEWM_HOME}/reacher"

echo "==> Preparing node-local copy in ${LOCAL_STABLEWM_HOME}"
rsync -a "${SHARED_DATASET_PATH}" "${LOCAL_STABLEWM_HOME}/reacher.h5"
rsync -a "${SHARED_CKPT_PATH}" "${LOCAL_STABLEWM_HOME}/reacher/lewm_object.ckpt"

export STABLEWM_HOME="${LOCAL_STABLEWM_HOME}"

######################################## TRAINING LAUNCH #######################################

RUN_NAME="${RUN_NAME:-hi_lewm_reacher_l2_d25_${SLURM_JOB_ID:-manual}}"
RUN_DIR="${SHARED_STABLEWM_HOME}/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

cd "${REPO_ROOT}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "Shared home: ${SHARED_STABLEWM_HOME}"
echo "Local home: ${LOCAL_STABLEWM_HOME}"
echo "STABLEWM_HOME (read path): ${STABLEWM_HOME}"
echo "Output run dir (shared): ${RUN_DIR}"
echo "TMPDIR: ${TMPDIR:-<unset>}"
echo "W&B entity: ${WANDB_ENTITY:-<default from login>}"
echo "W&B project: ${WANDB_PROJECT}"
echo "Run name: ${RUN_NAME}"
echo "Dataset: reacher.h5"
echo "Pretrained low-level checkpoint: ${LOCAL_STABLEWM_HOME}/reacher/lewm_object.ckpt"
echo "Launching: num_levels=2, k2=5 (d=25), data=hi_dmc, frozen pretrained low-level"

python hi_train.py \
  wm.num_levels=2 \
  wm.k1=0 \
  wm.k2=5 \
  data=hi_dmc \
  subdir="${RUN_DIR}" \
  pretrained_low_level.enabled=True \
  pretrained_low_level.checkpoint.path="${LOCAL_STABLEWM_HOME}/reacher/lewm_object.ckpt" \
  pretrained_low_level.freeze.encoder=True \
  pretrained_low_level.freeze.low_level_predictor=True \
  pretrained_low_level.freeze.low_level_action_encoder=True \
  pretrained_low_level.freeze.projector=True \
  pretrained_low_level.freeze.low_pred_proj=True \
  wandb.config.entity=${WANDB_ENTITY_OVERRIDE} \
  wandb.config.project="${WANDB_PROJECT}" \
  wandb.config.name="${RUN_NAME}" \
  wandb.config.id="run_${SLURM_JOB_ID:-manual}" \
  output_model_name="${RUN_NAME}"
