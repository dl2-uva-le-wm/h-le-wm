#!/usr/bin/env bash

# Submit the original PushT baseline matrix sweep as one Slurm array.
#
# Usage:
#   cd /gpfs/home2/scur0200/main/jobs/eval/original/pusht/matrix
#   ./submit_baseline_matrix.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARAMS_FILE="${PARAMS_FILE:-${SCRIPT_DIR}/baseline_matrix_sweep.csv}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval_baseline_matrix.sh}"
SWEEP_NAME="${SWEEP_NAME:-orig_pusht_cpu_matrix_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT_BASE="${LOG_ROOT_BASE:-${SCRIPT_DIR}/logs}"
LOG_ROOT="${LOG_ROOT:-${LOG_ROOT_BASE}/${SWEEP_NAME}}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_GPUS="${SBATCH_GPUS:-}"

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

if [[ ! -f "${PARAMS_FILE}" ]]; then
  echo "ERROR: parameter CSV not found: ${PARAMS_FILE}" >&2
  exit 2
fi

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
  exit 3
fi

NUM_CONFIGS="$(( $(awk 'NR > 1 && $0 !~ /^[[:space:]]*($|#)/ { count++ } END { print count + 0 }' "${PARAMS_FILE}") ))"
if ! [[ "${NUM_CONFIGS}" =~ ^[0-9]+$ ]] || (( NUM_CONFIGS <= 0 )); then
  echo "ERROR: parameter CSV is empty: ${PARAMS_FILE}" >&2
  exit 4
fi

if ! parse_seed_list "${EVAL_SEEDS:-42}"; then
  exit 5
fi
NUM_SEEDS="${#PARSED_SEEDS[@]}"
TOTAL_TASKS=$(( NUM_CONFIGS * NUM_SEEDS ))

mkdir -p "${LOG_ROOT}"

echo "Parameter CSV: ${PARAMS_FILE}"
echo "Job script: ${JOB_SCRIPT}"
echo "Sweep name: ${SWEEP_NAME}"
echo "Configs: ${NUM_CONFIGS}"
echo "Seeds: ${EVAL_SEEDS:-42}"
echo "Log root: ${LOG_ROOT}"
echo "Array: 1-${TOTAL_TASKS}"

SBATCH_CMD=(
  sbatch
  --array="1-${TOTAL_TASKS}"
  --output="${LOG_ROOT}/baseline_matrix_%A_%a.out"
  --error="${LOG_ROOT}/baseline_matrix_%A_%a.err"
  --export="ALL,SWEEP_NAME=${SWEEP_NAME},EVAL_SEEDS=${EVAL_SEEDS:-42},EVAL_DEVICE=${EVAL_DEVICE:-cpu},EVAL_DETERMINISM=${EVAL_DETERMINISM:-strict}"
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
