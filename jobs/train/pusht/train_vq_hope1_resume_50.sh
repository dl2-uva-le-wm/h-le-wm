#!/bin/bash

# Snellius resume job: continue the resumed PushT VQ high-level P2 run
# from epoch 33 to epoch 50.
#
# This wrapper pins the already-resumed run:
# - run: hi_lewm_p2_vq_train_hope1_22607223
# - previous completed epoch: 33
# - target max epochs: 50
#
# Usage:
#   cd jobs/train/pusht
#   sbatch train_vq_hope1_resume_50.sh
#
# Optional overrides:
#   MAX_EPOCHS=30 sbatch train_vq_hope1_resume_50.sh
#   MAX_EPOCHS=50 sbatch train_vq_hope1_resume_50.sh
#   SCRATCH_STABLEWM_HOME=/scratch-shared/$USER/stablewm_data sbatch train_vq_hope1_resume_50.sh

#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --job-name=hi_l2_pusht_vq_r50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=24:00:00
#SBATCH --output=train_vq_hope1_resume_50_%j.out
#SBATCH --error=train_vq_hope1_resume_50_%j.err

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

BASE_RESUME_SCRIPT="${REPO_ROOT}/jobs/train/pusht/train_vq_hope1_resume.sh"

if [[ ! -f "${BASE_RESUME_SCRIPT}" ]]; then
  echo "ERROR: base resume script not found: ${BASE_RESUME_SCRIPT}" >&2
  exit 2
fi

export PREVIOUS_JOB_ID="${PREVIOUS_JOB_ID:-22607223}"
export PREVIOUS_MAX_EPOCHS="${PREVIOUS_MAX_EPOCHS:-33}"
export MAX_EPOCHS="${MAX_EPOCHS:-50}"
export RESUME_RUN_NAME="${RESUME_RUN_NAME:-hi_lewm_p2_vq_train_hope1_${PREVIOUS_JOB_ID}}"
export RESUME_WANDB_RUN_ID="${RESUME_WANDB_RUN_ID:-run_${PREVIOUS_JOB_ID}}"

SCRATCH_STABLEWM_HOME="${SCRATCH_STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
export RESUME_OBJECT_CKPT="${RESUME_OBJECT_CKPT:-${SCRATCH_STABLEWM_HOME}/runs/${RESUME_RUN_NAME}/${RESUME_RUN_NAME}_epoch_${PREVIOUS_MAX_EPOCHS}_object.ckpt}"

if [[ ! -f "${RESUME_OBJECT_CKPT}" ]]; then
  echo "ERROR: expected resume object checkpoint not found: ${RESUME_OBJECT_CKPT}" >&2
  exit 2
fi

echo "Delegating to: ${BASE_RESUME_SCRIPT}"
echo "Resume run name: ${RESUME_RUN_NAME}"
echo "Previous completed epoch: ${PREVIOUS_MAX_EPOCHS}"
echo "Target max epochs: ${MAX_EPOCHS}"
echo "Expected object checkpoint anchor: ${RESUME_OBJECT_CKPT}"

exec bash "${BASE_RESUME_SCRIPT}"
