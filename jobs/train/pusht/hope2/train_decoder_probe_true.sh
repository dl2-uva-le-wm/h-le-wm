#!/usr/bin/env bash

# Convenience wrapper for Phase A of the frozen HOPE2 decoder probe.
# Trains the decoder only on true waypoint latents.
#
# Usage:
#   cd jobs/train/pusht
#   sbatch hope2/train_decoder_probe_true.sh
#   MAX_EPOCHS=10 sbatch hope2/train_decoder_probe_true.sh
#   TRAIN_RUN_NAME=hi_decoder_probe_true_custom sbatch hope2/train_decoder_probe_true.sh

#SBATCH --partition=gpu_h100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --job-name=hi_decoder_probe_true
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --output=output/hope2/train_decoder_probe_true_%j.out
#SBATCH --error=output/hope2/train_decoder_probe_true_%j.err

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

MODE=true_only
export MODE

if [[ -z "${TRAIN_RUN_NAME:-}" ]]; then
  export TRAIN_RUN_NAME="hi_decoder_probe_true_hope2_${SLURM_JOB_ID:-manual}"
fi

exec "${REPO_ROOT}/jobs/train/pusht/hope2/train_decoder_probe.sh"
