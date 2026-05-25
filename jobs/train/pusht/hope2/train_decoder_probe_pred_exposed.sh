#!/usr/bin/env bash

# Convenience wrapper for Phase B of the frozen HOPE2 decoder probe.
# Initializes from a Phase A decoder checkpoint and exposes the decoder to
# predicted HOPE2 waypoint latents without updating HOPE2 itself.
#
# Usage:
#   cd jobs/train/pusht
#   INIT_DECODER_CKPT=/scratch-shared/$USER/stablewm_data/runs/hi_decoder_probe_true_x/hi_decoder_probe_true_x_probe.pt \
#     sbatch hope2/train_decoder_probe_pred_exposed.sh
#
# Optional overrides:
#   MAX_EPOCHS=10 sbatch hope2/train_decoder_probe_pred_exposed.sh
#   TRAIN_RUN_NAME=hi_decoder_probe_pred_custom sbatch hope2/train_decoder_probe_pred_exposed.sh

#SBATCH --partition=gpu_h100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --job-name=hi_decoder_probe_pred
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --output=output/hope2/train_decoder_probe_pred_%j.out
#SBATCH --error=output/hope2/train_decoder_probe_pred_%j.err

set -euo pipefail

if [[ -z "${INIT_DECODER_CKPT:-}" ]]; then
  echo "ERROR: INIT_DECODER_CKPT must point to a Phase A decoder probe checkpoint." >&2
  exit 2
fi

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
        if [[ -f "${p}/jobs/train/pusht/hope2/train_decoder_probe.sh" ]]; then
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

MODE=pred_exposed
export MODE

if [[ -z "${TRAIN_RUN_NAME:-}" ]]; then
  export TRAIN_RUN_NAME="hi_decoder_probe_pred_exposed_hope2_${SLURM_JOB_ID:-manual}"
fi

exec "${REPO_ROOT}/jobs/train/pusht/hope2/train_decoder_probe.sh"
