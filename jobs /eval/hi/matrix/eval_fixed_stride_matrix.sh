#!/bin/bash

# Array-driven eval matrix for the fixed-stride PushT checkpoints.
#
# Matrix definition:
# - Checkpoints come from checkpoints_fixed_stride.txt
# - Planner configs are hardcoded in this script
# - One array task runs one checkpoint/config pair
#
# Current hardcoded configs per checkpoint:
#   1. d=25, high_h=1, low_h=2, low_receding_h=1
#   2. d=25, high_h=1, low_h=5, low_receding_h=1
#   3. d=25, high_h=2, low_h=2, low_receding_h=1
#   4. d=50, high_h=1, low_h=2, low_receding_h=1
#   5. d=50, high_h=2, low_h=2, low_receding_h=1
#   6. d=50, high_h=2, low_h=5, low_receding_h=1
#   7. d=50, high_h=2, low_h=5, low_receding_h=5
#
# Submit with:
#  cd /home/scur0200/main/jobs/eval/hi/matrix
#  ./submit_fixed_stride_matrix.sh




#SBATCH --partition=rome
#SBATCH --job-name=hi_eval_s5_matrix
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --chdir=/gpfs/home2/scur0200/main/jobs/eval/hi/matrix
#SBATCH --output=eval_fixed_stride_matrix_%A_%a.out
#SBATCH --error=eval_fixed_stride_matrix_%A_%a.err

set -euo pipefail

resolve_matrix_dir() {
  local c p
  for c in \
    "${MATRIX_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${PROJECT_ROOT:-}/jobs/eval/hi/matrix" \
    "/gpfs/home2/${USER}/main/jobs/eval/hi/matrix"; do
    [[ -z "${c}" ]] && continue
    if p="$(cd "${c}" >/dev/null 2>&1 && pwd)"; then
      if [[ -f "${p}/checkpoints_fixed_stride.txt" ]]; then
        echo "${p}"
        return 0
      fi
    fi
  done
  return 1
}

if ! MATRIX_DIR_RESOLVED="$(resolve_matrix_dir)"; then
  echo "ERROR: could not resolve matrix directory." >&2
  exit 2
fi

PROJECT_ROOT_DEFAULT="$(cd "${MATRIX_DIR_RESOLVED}/../../../.." >/dev/null 2>&1 && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

CHECKPOINT_FILE="${CHECKPOINT_FILE:-${MATRIX_DIR_RESOLVED}/checkpoints_fixed_stride.txt}"
BASE_SCRIPT="${BASE_SCRIPT:-${MATRIX_DIR_RESOLVED}/../hope2/hope2_pusht_eval_base.sh}"

if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
  echo "ERROR: checkpoint list not found: ${CHECKPOINT_FILE}" >&2
  exit 3
fi

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: base eval script not found: ${BASE_SCRIPT}" >&2
  exit 4
fi

mapfile -t CHECKPOINT_ROWS < <(grep -Ev '^[[:space:]]*($|#)' "${CHECKPOINT_FILE}")
NUM_CHECKPOINTS="${#CHECKPOINT_ROWS[@]}"
NUM_CONFIGS=7

if (( NUM_CHECKPOINTS == 0 )); then
  echo "ERROR: no checkpoint rows found in ${CHECKPOINT_FILE}" >&2
  exit 5
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set." >&2
  echo "Submit with ./submit_fixed_stride_matrix.sh or pass SLURM_ARRAY_TASK_ID manually." >&2
  exit 6
fi

if [[ -n "${CHECKPOINT_ROW_INDEX:-}" ]]; then
  if ! [[ "${CHECKPOINT_ROW_INDEX}" =~ ^[0-9]+$ ]] || (( CHECKPOINT_ROW_INDEX < 1 || CHECKPOINT_ROW_INDEX > NUM_CHECKPOINTS )); then
    echo "ERROR: CHECKPOINT_ROW_INDEX ${CHECKPOINT_ROW_INDEX} is out of range 1-${NUM_CHECKPOINTS}" >&2
    exit 7
  fi
  if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 1 || TASK_ID > NUM_CONFIGS )); then
    echo "ERROR: task id ${TASK_ID} is out of range 1-${NUM_CONFIGS} for per-checkpoint arrays" >&2
    exit 8
  fi
  CHECKPOINT_INDEX=$(( CHECKPOINT_ROW_INDEX - 1 ))
  CONFIG_INDEX="${TASK_ID}"
else
  if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 1 || TASK_ID > NUM_CHECKPOINTS * NUM_CONFIGS )); then
    echo "ERROR: task id ${TASK_ID} is out of range 1-$((NUM_CHECKPOINTS * NUM_CONFIGS))" >&2
    exit 8
  fi
  CHECKPOINT_INDEX=$(( (TASK_ID - 1) / NUM_CONFIGS ))
  CONFIG_INDEX=$(( (TASK_ID - 1) % NUM_CONFIGS + 1 ))
fi

read -r RUN_NAME CHECKPOINT_EPOCH <<< "${CHECKPOINT_ROWS[CHECKPOINT_INDEX]}"
if [[ -z "${RUN_NAME:-}" || -z "${CHECKPOINT_EPOCH:-}" ]]; then
  echo "ERROR: invalid checkpoint row: ${CHECKPOINT_ROWS[CHECKPOINT_INDEX]}" >&2
  exit 9
fi

RUN_SHORT="${RUN_NAME#hi_lewm_p2_train_}"
MODEL_LABEL="${RUN_SHORT}_ep${CHECKPOINT_EPOCH}"

export RUN_NAME
export CHECKPOINT_EPOCH
export MODEL_LABEL

export EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
export EVAL_BUDGET="${EVAL_BUDGET:-50}"

export HIGH_RECEDING_HORIZON=1
export HIGH_ACTION_BLOCK=1
export HIGH_REPLAN_INTERVAL=5

export LOW_ACTION_BLOCK=5

case "${CONFIG_INDEX}" in
  1)
    LABEL="d25_hh1_lh2_lrh1"
    export GOAL_OFFSET_STEPS=25
    export HIGH_HORIZON=1
    export LOW_HORIZON=2
    export LOW_RECEDING_HORIZON=1
    export HIGH_NUM_SAMPLES=900
    export HIGH_N_STEPS=20
    export HIGH_TOPK=10
    export LOW_NUM_SAMPLES=300
    export LOW_N_STEPS=30
    export LOW_TOPK=150
    ;;
  2)
    LABEL="d25_hh1_lh5_lrh1"
    export GOAL_OFFSET_STEPS=25
    export HIGH_HORIZON=1
    export LOW_HORIZON=5
    export LOW_RECEDING_HORIZON=1
    export HIGH_NUM_SAMPLES=900
    export HIGH_N_STEPS=20
    export HIGH_TOPK=10
    export LOW_NUM_SAMPLES=300
    export LOW_N_STEPS=30
    export LOW_TOPK=150
    ;;
  3)
    LABEL="d25_hh2_lh2_lrh1"
    export GOAL_OFFSET_STEPS=25
    export HIGH_HORIZON=2
    export LOW_HORIZON=2
    export LOW_RECEDING_HORIZON=1
    export HIGH_NUM_SAMPLES=900
    export HIGH_N_STEPS=20
    export HIGH_TOPK=10
    export LOW_NUM_SAMPLES=300
    export LOW_N_STEPS=30
    export LOW_TOPK=150
    ;;
  4)
    LABEL="d50_hh1_lh2_lrh1"
    export GOAL_OFFSET_STEPS=50
    export HIGH_HORIZON=1
    export LOW_HORIZON=2
    export LOW_RECEDING_HORIZON=1
    export HIGH_NUM_SAMPLES=1500
    export HIGH_N_STEPS=40
    export HIGH_TOPK=10
    export LOW_NUM_SAMPLES=900
    export LOW_N_STEPS=20
    export LOW_TOPK=150
    ;;
  5)
    LABEL="d50_hh2_lh2_lrh1"
    export GOAL_OFFSET_STEPS=50
    export HIGH_HORIZON=2
    export LOW_HORIZON=2
    export LOW_RECEDING_HORIZON=1
    export HIGH_NUM_SAMPLES=1500
    export HIGH_N_STEPS=40
    export HIGH_TOPK=10
    export LOW_NUM_SAMPLES=900
    export LOW_N_STEPS=20
    export LOW_TOPK=150
    ;;
  6)
    LABEL="d50_hh2_lh5_lrh1"
    export GOAL_OFFSET_STEPS=50
    export HIGH_HORIZON=2
    export LOW_HORIZON=5
    export LOW_RECEDING_HORIZON=1
    export HIGH_NUM_SAMPLES=1500
    export HIGH_N_STEPS=40
    export HIGH_TOPK=10
    export LOW_NUM_SAMPLES=900
    export LOW_N_STEPS=20
    export LOW_TOPK=150
    ;;
  7)
    LABEL="d50_hh2_lh5_lrh5"
    export GOAL_OFFSET_STEPS=50
    export HIGH_HORIZON=2
    export LOW_HORIZON=5
    export LOW_RECEDING_HORIZON=5
    export HIGH_NUM_SAMPLES=1500
    export HIGH_N_STEPS=40
    export HIGH_TOPK=10
    export LOW_NUM_SAMPLES=900
    export LOW_N_STEPS=20
    export LOW_TOPK=150
    ;;
  *)
    echo "ERROR: unsupported config index ${CONFIG_INDEX}" >&2
    exit 10
    ;;
esac

export EVAL_SUBDIR="eval_matrix_${LABEL}_job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-${TASK_ID}}"
export RESULT_FILENAME="${MODEL_LABEL}_${LABEL}_results.txt"

echo "Checkpoint row: $((CHECKPOINT_INDEX + 1)) / ${NUM_CHECKPOINTS}"
echo "Config row: ${CONFIG_INDEX} / ${NUM_CONFIGS}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint epoch: ${CHECKPOINT_EPOCH}"
echo "Label: ${LABEL}"
echo "Eval device: ${EVAL_DEVICE}"
echo "Matrix dir: ${MATRIX_DIR_RESOLVED}"
echo "Project root: ${PROJECT_ROOT}"
echo "Base script: ${BASE_SCRIPT}"
echo "Checkpoint list: ${CHECKPOINT_FILE}"

exec bash "${BASE_SCRIPT}"
