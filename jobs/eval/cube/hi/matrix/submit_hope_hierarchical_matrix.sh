#!/usr/bin/env bash

# Submit the full hope hierarchical matrix sweep.
#
# Run it with:
#   ./submit_hope_hierarchical_matrix.sh
#
# Override SWEEP_FILE to run a smaller ablation sweep instead.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-${SCRIPT_DIR}/checkpoints_hope_hierarchical.txt}"
SWEEP_FILE="${SWEEP_FILE:-${SCRIPT_DIR}/full_hierarchical_matrix_sweep.csv}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval_hope_hierarchical_matrix.sh}"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/run_hi_cube_matrix_eval.sh}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu_h100}"
SBATCH_GPUS="${SBATCH_GPUS:-1}"
SBATCH_ARRAY_SUFFIX="${SBATCH_ARRAY_SUFFIX:-}"

parse_seed_list() {
  local raw="${1:-42}"
  local cleaned="${raw//[[:space:]]/}"
  if [[ -z "${cleaned}" ]]; then
    echo "ERROR: EVAL_SEEDS is empty." >&2
    return 1
  fi

  IFS=',' read -r -a PARSED_SEEDS <<< "${cleaned}"
  if (( ${#PARSED_SEEDS[@]} == 0 )); then
    echo "ERROR: EVAL_SEEDS produced no seeds." >&2
    return 1
  fi

  for seed in "${PARSED_SEEDS[@]}"; do
    if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: invalid seed '${seed}' in EVAL_SEEDS='${raw}'." >&2
      return 1
    fi
  done
}

if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
  echo "ERROR: checkpoint list not found: ${CHECKPOINT_FILE}" >&2
  exit 2
fi

if [[ ! -f "${SWEEP_FILE}" ]]; then
  echo "ERROR: sweep file not found: ${SWEEP_FILE}" >&2
  exit 3
fi

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
  exit 4
fi

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: base eval script not found: ${BASE_SCRIPT}" >&2
  exit 5
fi

NUM_CHECKPOINTS="$(grep -Evc '^[[:space:]]*($|#)' "${CHECKPOINT_FILE}")"
NUM_CONFIGS="$(awk '
  BEGIN { count = 0 }
  NR == 1 { next }
  /^[[:space:]]*#/ { next }
  /^[[:space:]]*$/ { next }
  { count++ }
  END { print count }
' "${SWEEP_FILE}")"

if ! [[ "${NUM_CHECKPOINTS}" =~ ^[0-9]+$ ]] || (( NUM_CHECKPOINTS <= 0 )); then
  echo "ERROR: checkpoint list is empty: ${CHECKPOINT_FILE}" >&2
  exit 6
fi

if ! [[ "${NUM_CONFIGS}" =~ ^[0-9]+$ ]] || (( NUM_CONFIGS <= 0 )); then
  echo "ERROR: sweep is empty: ${SWEEP_FILE}" >&2
  exit 7
fi

if ! parse_seed_list "${EVAL_SEEDS:-42}"; then
  exit 8
fi
NUM_SEEDS="${#PARSED_SEEDS[@]}"
TOTAL_ARRAY_TASKS=$(( NUM_CONFIGS * NUM_SEEDS ))
ARRAY_SPEC="1-${TOTAL_ARRAY_TASKS}${SBATCH_ARRAY_SUFFIX}"

mapfile -t CHECKPOINT_ROWS < <(grep -Ev '^[[:space:]]*($|#)' "${CHECKPOINT_FILE}")
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs_hope_hierarchical_matrix}"
mkdir -p "${LOG_ROOT}"

echo "Checkpoint file: ${CHECKPOINT_FILE}"
echo "Sweep file: ${SWEEP_FILE}"
echo "Job script: ${JOB_SCRIPT}"
echo "Base script: ${BASE_SCRIPT}"
echo "Requested eval device: ${EVAL_DEVICE:-cuda}"
echo "Requested partition: ${SBATCH_PARTITION:-<script default>}"
echo "Requested gpus: ${SBATCH_GPUS:-<script default>}"
echo "Checkpoints: ${NUM_CHECKPOINTS}"
echo "Configs per checkpoint: ${NUM_CONFIGS}"
echo "Seeds: ${EVAL_SEEDS:-42}"
echo "Tasks per checkpoint array: ${TOTAL_ARRAY_TASKS}"
echo "Array spec: ${ARRAY_SPEC}"
echo "Log root: ${LOG_ROOT}"

for i in "${!CHECKPOINT_ROWS[@]}"; do
  read -r RUN_NAME CHECKPOINT_EPOCH <<< "${CHECKPOINT_ROWS[i]}"
  LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
  mkdir -p "${LOG_DIR}"

  echo
  echo "Submitting checkpoint $((i + 1))/${NUM_CHECKPOINTS}: ${RUN_NAME} (epoch ${CHECKPOINT_EPOCH})"
  echo "Log dir: ${LOG_DIR}"

  SBATCH_CMD=(
    sbatch
    --array="${ARRAY_SPEC}"
    --output="${LOG_DIR}/eval_hope_hierarchical_matrix_%A_%a.out"
    --error="${LOG_DIR}/eval_hope_hierarchical_matrix_%A_%a.err"
    --export="ALL,CHECKPOINT_ROW_INDEX=$((i + 1)),BASE_SCRIPT=${BASE_SCRIPT},EVAL_DEVICE=${EVAL_DEVICE:-cuda}"
  )

  if [[ -n "${SBATCH_PARTITION}" ]]; then
    SBATCH_CMD+=(--partition="${SBATCH_PARTITION}")
  fi

  if [[ -n "${SBATCH_GPUS}" ]]; then
    SBATCH_CMD+=(--gpus="${SBATCH_GPUS}")
  fi

  SBATCH_CMD+=("${JOB_SCRIPT}")

  printf 'Submitting with:'
  printf ' %q' "${SBATCH_CMD[@]}"
  echo

  "${SBATCH_CMD[@]}"
done
