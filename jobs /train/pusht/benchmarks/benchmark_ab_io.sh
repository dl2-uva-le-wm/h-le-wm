#!/bin/bash

# Snellius A/B benchmark: compare shared scratch vs node-local TMPDIR dataset I/O.
# Usage:
#   cd jobs/train/pusht
#   sbatch benchmark_ab_io.sh
#
# Optional overrides:
#   BENCH_STEPS=500 BENCH_LIMIT_VAL_BATCHES=0 sbatch benchmark_ab_io.sh
#   SCRATCH_STABLEWM_HOME=/scratch-shared/$USER/stablewm_data sbatch benchmark_ab_io.sh
#   BENCH_RESUME=1 BENCH_RUN_NAME_SHARED=hi_lewm_p2_bench_shared_21983090 sbatch benchmark_ab_io.sh

#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --job-name=hi_l2_pusht_ab_io
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=03:00:00
#SBATCH --output=bench_ab_io_%j.out
#SBATCH --error=bench_ab_io_%j.err

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

if [[ -z "${TMPDIR:-}" ]]; then
  echo "ERROR: TMPDIR is not set." >&2
  echo "Expected a scratch-node allocation where TMPDIR points under /scratch-node." >&2
  exit 2
fi
if [[ "${TMPDIR}" != /scratch-node/* ]]; then
  echo "ERROR: TMPDIR is '${TMPDIR}', expected /scratch-node/... for a fair NVMe benchmark." >&2
  echo "Make sure this job is submitted with '#SBATCH --constraint=scratch-node'." >&2
  exit 2
fi

BENCH_STEPS="${BENCH_STEPS:-500}"
BENCH_LIMIT_VAL_BATCHES="${BENCH_LIMIT_VAL_BATCHES:-0}"
BENCH_MAX_EPOCHS="${BENCH_MAX_EPOCHS:-9999}"
BENCH_RESUME="${BENCH_RESUME:-0}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
BENCH_RUN_NAME_SHARED="${BENCH_RUN_NAME_SHARED:-hi_lewm_p2_bench_shared_${RUN_TAG}}"
BENCH_RUN_NAME_LOCAL="${BENCH_RUN_NAME_LOCAL:-hi_lewm_p2_bench_local_${RUN_TAG}}"

LOCAL_STABLEWM_HOME="${LOCAL_STABLEWM_HOME:-${TMPDIR:-/tmp}/${USER}_stablewm_data_${SLURM_JOB_ID:-manual}}"
LOCAL_DATASET="${LOCAL_STABLEWM_HOME}/${DATASET_FILE}"
LOCAL_CKPT="${LOCAL_STABLEWM_HOME}/${CKPT_REL}"

LOG_DIR="${REPO_ROOT}/jobs/train/pusht/out"
mkdir -p "${LOG_DIR}"
SHARED_LOG="${LOG_DIR}/ab_io_shared_${SLURM_JOB_ID:-manual}.log"
LOCAL_LOG="${LOG_DIR}/ab_io_local_${SLURM_JOB_ID:-manual}.log"

echo "Repo root: ${REPO_ROOT}"
echo "Scratch home: ${SCRATCH_STABLEWM_HOME}"
echo "Local home: ${LOCAL_STABLEWM_HOME}"
echo "Dataset: ${DATASET_FILE}"
echo "Checkpoint: ${CKPT_REL}"
echo "Bench steps: ${BENCH_STEPS}"
echo "Bench resume mode: ${BENCH_RESUME} (0=fresh, 1=resume)"
echo "Shared run name: ${BENCH_RUN_NAME_SHARED}"
echo "Local run name: ${BENCH_RUN_NAME_LOCAL}"

echo ""
echo "==> Preparing node-local copy in ${LOCAL_STABLEWM_HOME}"
mkdir -p "$(dirname "${LOCAL_DATASET}")" "$(dirname "${LOCAL_CKPT}")"
rsync -ah --info=progress2 "${SRC_DATASET}" "${LOCAL_DATASET}"
rsync -ah --info=progress2 "${SRC_CKPT}" "${LOCAL_CKPT}"

echo ""
echo "==> Run A: scratch-shared benchmark"
(
  export STABLEWM_HOME="${SCRATCH_STABLEWM_HOME}"
  export PRETRAINED_LEWM_CKPT="${SRC_CKPT}"
  export BENCH_STEPS="${BENCH_STEPS}"
  export BENCH_LIMIT_VAL_BATCHES="${BENCH_LIMIT_VAL_BATCHES}"
  export BENCH_MAX_EPOCHS="${BENCH_MAX_EPOCHS}"
  export BENCH_RUN_NAME="${BENCH_RUN_NAME_SHARED}"
  export BENCH_RESUME="${BENCH_RESUME}"
  bash "${BASE_BENCH}"
) | tee "${SHARED_LOG}"

echo ""
echo "==> Run B: node-local benchmark"
(
  export STABLEWM_HOME="${LOCAL_STABLEWM_HOME}"
  export PRETRAINED_LEWM_CKPT="${LOCAL_CKPT}"
  export BENCH_STEPS="${BENCH_STEPS}"
  export BENCH_LIMIT_VAL_BATCHES="${BENCH_LIMIT_VAL_BATCHES}"
  export BENCH_MAX_EPOCHS="${BENCH_MAX_EPOCHS}"
  export BENCH_RUN_NAME="${BENCH_RUN_NAME_LOCAL}"
  export BENCH_RESUME="${BENCH_RESUME}"
  bash "${BASE_BENCH}"
) | tee "${LOCAL_LOG}"

shared_s_per_step="$(grep -Eo '[0-9]+\.[0-9]+ s/step' "${SHARED_LOG}" | tail -n1 | awk '{print $1}')"
local_s_per_step="$(grep -Eo '[0-9]+\.[0-9]+ s/step' "${LOCAL_LOG}" | tail -n1 | awk '{print $1}')"

echo ""
echo "==> A/B Summary"
echo "scratch-shared s/step: ${shared_s_per_step:-N/A}"
echo "node-local    s/step: ${local_s_per_step:-N/A}"

if [[ -n "${shared_s_per_step:-}" && -n "${local_s_per_step:-}" ]]; then
  python - <<PY
shared = float("${shared_s_per_step}")
local = float("${local_s_per_step}")
improvement_pct = (shared - local) / shared * 100.0
speedup = shared / local
print(f"speedup (shared/local): {speedup:.3f}x")
print(f"relative improvement: {improvement_pct:.1f}%")
PY
else
  echo "Could not parse one or both s/step values from logs."
fi

echo "Shared log: ${SHARED_LOG}"
echo "Local log:  ${LOCAL_LOG}"
