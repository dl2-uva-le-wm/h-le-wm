#!/bin/bash

# Snellius single benchmark: P2 training with node-local storage on scratch-node.
# This is equivalent to the "Run B" (node-local) path from benchmark_ab_io.sh.
# Usage:
#   cd jobs/train/pusht
#   sbatch benchmark_cpu_optimization.sh
#
# Optional overrides:
#   BENCH_STEPS=500 BENCH_LIMIT_VAL_BATCHES=0 sbatch benchmark_cpu_optimization.sh
#   SCRATCH_STABLEWM_HOME=/scratch-shared/$USER/stablewm_data sbatch benchmark_cpu_optimization.sh

#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --job-name=hi_l2_pusht_cpu_opt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=bench_cpu_opt_%j.out
#SBATCH --error=bench_cpu_opt_%j.err

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

BASE_BENCH="${REPO_ROOT}/jobs/train/pusht/benchmark.sh"
if [[ ! -f "${BASE_BENCH}" ]]; then
  # Backward-compatible fallback for older branch layouts.
  BASE_BENCH="${REPO_ROOT}/jobs/2_levels/pusht/benchmark.sh"
fi
if [[ ! -f "${BASE_BENCH}" ]]; then
  echo "ERROR: benchmark script not found: ${BASE_BENCH}" >&2
  exit 2
fi

if [[ -z "${TMPDIR:-}" ]]; then
  echo "ERROR: TMPDIR is not set." >&2
  echo "Expected a scratch-node allocation where TMPDIR points under /scratch-node." >&2
  exit 2
fi
if [[ "${TMPDIR}" != /scratch-node/* ]]; then
  echo "ERROR: TMPDIR is '${TMPDIR}', expected /scratch-node/... for node-local benchmark." >&2
  echo "Make sure this job is submitted with '#SBATCH --constraint=scratch-node'." >&2
  exit 2
fi

SCRATCH_STABLEWM_HOME="${SCRATCH_STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
DATASET_FILE="${DATASET_FILE:-pusht_expert_train.h5}"
CKPT_REL="${CKPT_REL:-pusht/lewm_object.ckpt}"

SRC_DATASET="${SCRATCH_STABLEWM_HOME}/${DATASET_FILE}"
SRC_CKPT="${SCRATCH_STABLEWM_HOME}/${CKPT_REL}"
if [[ ! -f "${SRC_DATASET}" ]]; then
  echo "ERROR: dataset file not found: ${SRC_DATASET}" >&2
  exit 2
fi
if [[ ! -f "${SRC_CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${SRC_CKPT}" >&2
  exit 2
fi

BENCH_STEPS="${BENCH_STEPS:-500}"
BENCH_LIMIT_VAL_BATCHES="${BENCH_LIMIT_VAL_BATCHES:-0}"
BENCH_MAX_EPOCHS="${BENCH_MAX_EPOCHS:-9999}"
BENCH_RUN_NAME="${BENCH_RUN_NAME:-hi_lewm_p2_bench_local_cpu_opt}"

LOCAL_STABLEWM_HOME="${LOCAL_STABLEWM_HOME:-${TMPDIR}/${USER}_stablewm_data_${SLURM_JOB_ID:-manual}}"
LOCAL_DATASET="${LOCAL_STABLEWM_HOME}/${DATASET_FILE}"
LOCAL_CKPT="${LOCAL_STABLEWM_HOME}/${CKPT_REL}"

LOG_DIR="${REPO_ROOT}/jobs/train/pusht/out"
mkdir -p "${LOG_DIR}"
LOCAL_LOG="${LOG_DIR}/cpu_opt_local_${SLURM_JOB_ID:-manual}.log"

echo "Repo root: ${REPO_ROOT}"
echo "Scratch home: ${SCRATCH_STABLEWM_HOME}"
echo "Local home: ${LOCAL_STABLEWM_HOME}"
echo "TMPDIR: ${TMPDIR}"
echo "Dataset: ${DATASET_FILE}"
echo "Checkpoint: ${CKPT_REL}"
echo "Bench steps: ${BENCH_STEPS}"
echo "Bench run name: ${BENCH_RUN_NAME}"

echo ""
echo "==> Preparing node-local copy in ${LOCAL_STABLEWM_HOME}"
mkdir -p "$(dirname "${LOCAL_DATASET}")" "$(dirname "${LOCAL_CKPT}")"
rsync -ah --info=progress2 "${SRC_DATASET}" "${LOCAL_DATASET}"
rsync -ah --info=progress2 "${SRC_CKPT}" "${LOCAL_CKPT}"

echo ""
echo "==> Running node-local benchmark (CPU-optimization build)"
(
  export STABLEWM_HOME="${LOCAL_STABLEWM_HOME}"
  export PRETRAINED_LEWM_CKPT="${LOCAL_CKPT}"
  export BENCH_STEPS="${BENCH_STEPS}"
  export BENCH_LIMIT_VAL_BATCHES="${BENCH_LIMIT_VAL_BATCHES}"
  export BENCH_MAX_EPOCHS="${BENCH_MAX_EPOCHS}"
  export BENCH_RUN_NAME="${BENCH_RUN_NAME}"
  bash "${BASE_BENCH}"
) | tee "${LOCAL_LOG}"

local_s_per_step="$(grep -Eo '[0-9]+\.[0-9]+ s/step' "${LOCAL_LOG}" | tail -n1 | awk '{print $1}' || true)"

echo ""
echo "==> Summary"
echo "node-local s/step: ${local_s_per_step:-N/A}"
if [[ -z "${local_s_per_step:-}" ]]; then
  echo "Could not parse s/step from log."
fi
echo "Log: ${LOCAL_LOG}"
