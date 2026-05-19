#!/bin/bash

# Run it with:
# ./submit_fixed_stride_matrix.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-${SCRIPT_DIR}/checkpoints_fixed_stride.txt}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval_fixed_stride_matrix.sh}"

if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
  echo "ERROR: checkpoint list not found: ${CHECKPOINT_FILE}" >&2
  exit 2
fi

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
  exit 3
fi

NUM_CHECKPOINTS="$(grep -Evc '^[[:space:]]*($|#)' "${CHECKPOINT_FILE}")"
if ! [[ "${NUM_CHECKPOINTS}" =~ ^[0-9]+$ ]] || (( NUM_CHECKPOINTS <= 0 )); then
  echo "ERROR: checkpoint list is empty: ${CHECKPOINT_FILE}" >&2
  exit 4
fi

mapfile -t CHECKPOINT_ROWS < <(grep -Ev '^[[:space:]]*($|#)' "${CHECKPOINT_FILE}")
NUM_CONFIGS=7
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_ROOT}"

echo "Checkpoint file: ${CHECKPOINT_FILE}"
echo "Job script: ${JOB_SCRIPT}"
echo "Checkpoints: ${NUM_CHECKPOINTS}"
echo "Configs per checkpoint: ${NUM_CONFIGS}"
echo "Log root: ${LOG_ROOT}"

for i in "${!CHECKPOINT_ROWS[@]}"; do
  read -r RUN_NAME CHECKPOINT_EPOCH <<< "${CHECKPOINT_ROWS[i]}"
  LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
  mkdir -p "${LOG_DIR}"

  echo ""
  echo "Submitting checkpoint $((i + 1))/${NUM_CHECKPOINTS}: ${RUN_NAME} (epoch ${CHECKPOINT_EPOCH})"
  echo "Log dir: ${LOG_DIR}"

  sbatch \
    --array="1-${NUM_CONFIGS}" \
    --output="${LOG_DIR}/eval_fixed_stride_matrix_%A_%a.out" \
    --error="${LOG_DIR}/eval_fixed_stride_matrix_%A_%a.err" \
    --export="ALL,CHECKPOINT_ROW_INDEX=$((i + 1))" \
    "${JOB_SCRIPT}"
done
