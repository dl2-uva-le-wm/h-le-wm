#!/bin/bash

# Snellius resume job: continue the latest completed PushT VQ high-level P2 run.
# This wrapper reuses the exact run directory and W&B id from job 22607223 so
# hi_train.py resumes from the existing Lightning checkpoint instead of starting fresh.
#
# Usage:
#   cd jobs/train/pusht
#   sbatch train_vq_hope1_resume.sh
#
# Optional overrides:
#   MAX_EPOCHS=20 sbatch train_vq_hope1_resume.sh
#   MAX_EPOCHS=30 sbatch train_vq_hope1_resume.sh
#   RESUME_RUN_NAME=hi_lewm_p2_vq_train_hope1_22607223 sbatch train_vq_hope1_resume.sh
#   SCRATCH_STABLEWM_HOME=/scratch-shared/$USER/stablewm_data sbatch train_vq_hope1_resume.sh

#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --job-name=hi_l2_pusht_vq_resume
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=07:00:00
#SBATCH --output=train_vq_hope1_resume_%j.out
#SBATCH --error=train_vq_hope1_resume_%j.err

set -euo pipefail

resolve_repo_root() {
  local c p
  for c in \
    "${PROJECT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${HOME}/h-le-wm" \
    "${HOME}/h-lewm" \
    "/gpfs/home2/${USER}/h-le-wm" \
    "/gpfs/home2/${USER}/h-lewm"; do
    [[ -z "${c}" ]] && continue
    for p in "${c}" "${c}/.." "${c}/../.." "${c}/../../.."; do
      if p="$(cd "${p}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${p}/hi_train.py" && -f "${p}/config/train/hi_lewm.yaml" ]]; then
          echo "${p}"
          return 0
        fi
      fi
    done
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root." >&2
  exit 2
fi

BASE_TRAIN_SCRIPT="${REPO_ROOT}/jobs/train/pusht/train_vq_hope1.sh"

if [[ ! -f "${BASE_TRAIN_SCRIPT}" ]]; then
  echo "ERROR: base train script not found: ${BASE_TRAIN_SCRIPT}" >&2
  exit 2
fi

SCRATCH_STABLEWM_HOME="${SCRATCH_STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
PREVIOUS_JOB_ID="${PREVIOUS_JOB_ID:-22607223}"
PREVIOUS_MAX_EPOCHS="${PREVIOUS_MAX_EPOCHS:-10}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"

RESUME_RUN_NAME="${RESUME_RUN_NAME:-hi_lewm_p2_vq_train_hope1_${PREVIOUS_JOB_ID}}"
RESUME_WANDB_RUN_ID="${RESUME_WANDB_RUN_ID:-run_${PREVIOUS_JOB_ID}}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-${SCRATCH_STABLEWM_HOME}/runs/${RESUME_RUN_NAME}}"
RESUME_WEIGHTS_CKPT="${RESUME_WEIGHTS_CKPT:-${RESUME_RUN_DIR}/${RESUME_RUN_NAME}_weights.ckpt}"
RESUME_OBJECT_CKPT="${RESUME_OBJECT_CKPT:-${RESUME_RUN_DIR}/${RESUME_RUN_NAME}_epoch_${PREVIOUS_MAX_EPOCHS}_object.ckpt}"

if ! [[ "${MAX_EPOCHS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MAX_EPOCHS must be a positive integer (got '${MAX_EPOCHS}')." >&2
  exit 2
fi
if [[ "${MAX_EPOCHS}" -le "${PREVIOUS_MAX_EPOCHS}" ]]; then
  echo "ERROR: MAX_EPOCHS must be greater than ${PREVIOUS_MAX_EPOCHS} to continue training (got '${MAX_EPOCHS}')." >&2
  exit 2
fi

if [[ ! -d "${RESUME_RUN_DIR}" ]]; then
  echo "ERROR: resume run directory not found: ${RESUME_RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${RESUME_WEIGHTS_CKPT}" ]]; then
  echo "ERROR: resume weights checkpoint not found: ${RESUME_WEIGHTS_CKPT}" >&2
  echo "Expected the Lightning checkpoint created by the previous run." >&2
  exit 2
fi

echo "Previous Slurm job: ${PREVIOUS_JOB_ID}"
echo "Scratch home: ${SCRATCH_STABLEWM_HOME}"
echo "Resume run name: ${RESUME_RUN_NAME}"
echo "Resume W&B run id: ${RESUME_WANDB_RUN_ID}"
echo "Resume run dir: ${RESUME_RUN_DIR}"
echo "Resume weights checkpoint: ${RESUME_WEIGHTS_CKPT}"
if [[ -f "${RESUME_OBJECT_CKPT}" ]]; then
  echo "Latest object checkpoint: ${RESUME_OBJECT_CKPT}"
fi
echo "Previous max epochs: ${PREVIOUS_MAX_EPOCHS}"
echo "Target max epochs: ${MAX_EPOCHS}"
echo "Delegating to: ${BASE_TRAIN_SCRIPT}"

export TRAIN_RUN_NAME="${RESUME_RUN_NAME}"
export WANDB_RUN_ID="${RESUME_WANDB_RUN_ID}"
export PERSIST_RUN_DIR="${RESUME_RUN_DIR}"
export MAX_EPOCHS

exec bash "${BASE_TRAIN_SCRIPT}"
