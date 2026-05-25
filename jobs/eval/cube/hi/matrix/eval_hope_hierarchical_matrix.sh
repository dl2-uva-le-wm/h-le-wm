#!/usr/bin/env bash

# Array-driven hope hierarchical matrix sweep for Cube checkpoints.
#
# Sweep source:
# - Config rows come from full_hierarchical_matrix_sweep.csv by default
# - Checkpoints come from checkpoints_hope_hierarchical.txt
#
# Config notes:
# - config_note blank     => planning.mode=hierarchical
# - config_note hiStaged => planning.mode=hierarchical_staged
# - eval_budget varies by row, but num_eval is pinned to 50 for every run
# - Defaults to CPU, but EVAL_DEVICE may be overridden at submission time
#   for targeted GPU reruns. The shared base launcher forwards solver.device
#   overrides and clears CUDA visibility only when cpu mode is selected.
#
# Submit with:
#   cd /gpfs/home2/scur0200/main/jobs/eval/cube/hi/matrix
#   ./submit_hope_hierarchical_matrix.sh

#SBATCH --partition=rome
#SBATCH --job-name=hi_eval_hope_matrix
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --chdir=/gpfs/home2/scur0200/main/jobs/eval/cube/hi/matrix
#SBATCH --output=eval_hope_hierarchical_matrix_%A_%a.out
#SBATCH --error=eval_hope_hierarchical_matrix_%A_%a.err

set -euo pipefail

resolve_matrix_dir() {
  local c p
  for c in \
    "${MATRIX_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${PROJECT_ROOT:-}/jobs/eval/cube/hi/matrix" \
    "/gpfs/home2/${USER}/main/jobs/eval/cube/hi/matrix"; do
    [[ -z "${c}" ]] && continue
    if p="$(cd "${c}" >/dev/null 2>&1 && pwd)"; then
      if [[ -f "${p}/eval_hope_hierarchical_matrix.sh" && -f "${p}/checkpoints_hope_hierarchical.txt" ]]; then
        echo "${p}"
        return 0
      fi
    fi
  done
  return 1
}

normalize_mode() {
  case "$1" in
    "" )
      echo "hierarchical"
      ;;
    hiStaged )
      echo "hierarchical_staged"
      ;;
    * )
      return 1
      ;;
  esac
}

goal_offset_steps_from_tag() {
  case "$1" in
    D25) echo "25" ;;
    D50) echo "50" ;;
    D75) echo "75" ;;
    * ) return 1 ;;
  esac
}

goal_slug_from_tag() {
  case "$1" in
    D25) echo "d25" ;;
    D50) echo "d50" ;;
    D75) echo "d75" ;;
    * ) return 1 ;;
  esac
}

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

if ! MATRIX_DIR_RESOLVED="$(resolve_matrix_dir)"; then
  echo "ERROR: could not resolve matrix directory." >&2
  exit 2
fi

PROJECT_ROOT_DEFAULT="$(cd "${MATRIX_DIR_RESOLVED}/../../../.." >/dev/null 2>&1 && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

CHECKPOINT_FILE="${CHECKPOINT_FILE:-${MATRIX_DIR_RESOLVED}/checkpoints_hope_hierarchical.txt}"
SWEEP_FILE="${SWEEP_FILE:-${MATRIX_DIR_RESOLVED}/full_hierarchical_matrix_sweep.csv}"
BASE_SCRIPT="${BASE_SCRIPT:-${MATRIX_DIR_RESOLVED}/run_hi_cube_matrix_eval.sh}"

if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
  echo "ERROR: checkpoint list not found: ${CHECKPOINT_FILE}" >&2
  exit 3
fi

if [[ ! -f "${SWEEP_FILE}" ]]; then
  echo "ERROR: sweep file not found: ${SWEEP_FILE}" >&2
  exit 4
fi

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: base eval script not found: ${BASE_SCRIPT}" >&2
  exit 5
fi

mapfile -t CHECKPOINT_ROWS < <(grep -Ev '^[[:space:]]*($|#)' "${CHECKPOINT_FILE}")
mapfile -t SWEEP_ROWS < <(awk '
  BEGIN { FS=";" }
  NR == 1 { next }
  /^[[:space:]]*#/ { next }
  /^[[:space:]]*$/ { next }
  {
    gsub(/\r/, "")
    print
  }
' "${SWEEP_FILE}")

NUM_CHECKPOINTS="${#CHECKPOINT_ROWS[@]}"
NUM_CONFIGS="${#SWEEP_ROWS[@]}"

if ! parse_seed_list "${EVAL_SEEDS:-42}"; then
  exit 8
fi
NUM_SEEDS="${#PARSED_SEEDS[@]}"

if (( NUM_CHECKPOINTS == 0 )); then
  echo "ERROR: no checkpoint rows found in ${CHECKPOINT_FILE}" >&2
  exit 9
fi

if (( NUM_CONFIGS == 0 )); then
  echo "ERROR: no sweep rows found in ${SWEEP_FILE}" >&2
  exit 10
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set." >&2
  echo "Submit with ./submit_hope_hierarchical_matrix.sh or pass SLURM_ARRAY_TASK_ID manually." >&2
  exit 11
fi

if [[ -n "${CHECKPOINT_ROW_INDEX:-}" ]]; then
  if ! [[ "${CHECKPOINT_ROW_INDEX}" =~ ^[0-9]+$ ]] || (( CHECKPOINT_ROW_INDEX < 1 || CHECKPOINT_ROW_INDEX > NUM_CHECKPOINTS )); then
    echo "ERROR: CHECKPOINT_ROW_INDEX ${CHECKPOINT_ROW_INDEX} is out of range 1-${NUM_CHECKPOINTS}" >&2
    exit 12
  fi
  TOTAL_TASKS=$(( NUM_CONFIGS * NUM_SEEDS ))
  if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 1 || TASK_ID > TOTAL_TASKS )); then
    echo "ERROR: task id ${TASK_ID} is out of range 1-${TOTAL_TASKS} for per-checkpoint arrays" >&2
    exit 13
  fi
  CHECKPOINT_INDEX=$(( CHECKPOINT_ROW_INDEX - 1 ))
  CONFIG_INDEX=$(( (TASK_ID - 1) / NUM_SEEDS + 1 ))
  SEED_INDEX=$(( (TASK_ID - 1) % NUM_SEEDS ))
else
  TASKS_PER_CHECKPOINT=$(( NUM_CONFIGS * NUM_SEEDS ))
  TOTAL_TASKS=$(( NUM_CHECKPOINTS * TASKS_PER_CHECKPOINT ))
  if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 1 || TASK_ID > TOTAL_TASKS )); then
    echo "ERROR: task id ${TASK_ID} is out of range 1-${TOTAL_TASKS}" >&2
    exit 14
  fi
  CHECKPOINT_INDEX=$(( (TASK_ID - 1) / TASKS_PER_CHECKPOINT ))
  TASK_ID_WITHIN_CHECKPOINT=$(( (TASK_ID - 1) % TASKS_PER_CHECKPOINT ))
  CONFIG_INDEX=$(( TASK_ID_WITHIN_CHECKPOINT / NUM_SEEDS + 1 ))
  SEED_INDEX=$(( TASK_ID_WITHIN_CHECKPOINT % NUM_SEEDS ))
fi

EVAL_SEED_VALUE="${PARSED_SEEDS[SEED_INDEX]}"

read -r RUN_NAME CHECKPOINT_EPOCH <<< "${CHECKPOINT_ROWS[CHECKPOINT_INDEX]}"
if [[ -z "${RUN_NAME:-}" || -z "${CHECKPOINT_EPOCH:-}" ]]; then
  echo "ERROR: invalid checkpoint row: ${CHECKPOINT_ROWS[CHECKPOINT_INDEX]}" >&2
  exit 15
fi

IFS=';' read -r GOAL_OFFSET_TAG CONFIG_NOTE EVAL_BUDGET_CSV HIGH_HORIZON_CSV LOW_HORIZON_CSV HIGH_RECEDING_HORIZON_CSV LOW_RECEDING_HORIZON_CSV HIGH_REPLAN_INTERVAL_CSV HIGH_ACTION_BLOCK_CSV LOW_ACTION_BLOCK_CSV HIGH_NUM_SAMPLES_CSV HIGH_N_STEPS_CSV HIGH_TOPK_CSV LOW_NUM_SAMPLES_CSV LOW_N_STEPS_CSV LOW_TOPK_CSV <<< "${SWEEP_ROWS[CONFIG_INDEX - 1]}"

if ! GOAL_OFFSET_STEPS_VALUE="$(goal_offset_steps_from_tag "${GOAL_OFFSET_TAG}")"; then
  echo "ERROR: unsupported goal_offset tag '${GOAL_OFFSET_TAG}' in row ${CONFIG_INDEX}" >&2
  exit 16
fi

if ! GOAL_SLUG="$(goal_slug_from_tag "${GOAL_OFFSET_TAG}")"; then
  echo "ERROR: unsupported goal_offset tag '${GOAL_OFFSET_TAG}' in row ${CONFIG_INDEX}" >&2
  exit 17
fi

if ! PLANNING_MODE_VALUE="$(normalize_mode "${CONFIG_NOTE}")"; then
  echo "ERROR: unsupported config_note '${CONFIG_NOTE}' in row ${CONFIG_INDEX}" >&2
  echo "Use blank or hiStaged." >&2
  exit 18
fi

MODE_TAG="hier"
if [[ "${PLANNING_MODE_VALUE}" == "hierarchical_staged" ]]; then
  MODE_TAG="staged"
fi

RUN_SHORT="${RUN_NAME#hi_lewm_p2_train_}"
RUN_SHORT="${RUN_SHORT#hi_lewm_cube_train_}"
MODEL_LABEL="${RUN_SHORT}_ep${CHECKPOINT_EPOCH}"
LABEL="cfg$(printf '%02d' "${CONFIG_INDEX}")_${GOAL_SLUG}_${MODE_TAG}_hh${HIGH_HORIZON_CSV}_lh${LOW_HORIZON_CSV}_b${EVAL_BUDGET_CSV}"
DISPLAY_LABEL="${LABEL}"
if (( NUM_SEEDS > 1 )); then
  DISPLAY_LABEL="${LABEL}_seed${EVAL_SEED_VALUE}"
fi

export RUN_NAME
export CHECKPOINT_EPOCH
export MODEL_LABEL
export CONFIG_NAME="${CONFIG_NAME:-hi_cube}"
export PLANNING_MODE="${PLANNING_MODE_VALUE}"
export NUM_EVAL=50
export EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
export GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS_VALUE}"
export EVAL_BUDGET="${EVAL_BUDGET_CSV}"

export HIGH_HORIZON="${HIGH_HORIZON_CSV}"
export LOW_HORIZON="${LOW_HORIZON_CSV}"
export HIGH_RECEDING_HORIZON="${HIGH_RECEDING_HORIZON_CSV}"
export LOW_RECEDING_HORIZON="${LOW_RECEDING_HORIZON_CSV}"
export HIGH_REPLAN_INTERVAL="${HIGH_REPLAN_INTERVAL_CSV}"
export HIGH_ACTION_BLOCK="${HIGH_ACTION_BLOCK_CSV}"
export LOW_ACTION_BLOCK="${LOW_ACTION_BLOCK_CSV}"
export HIGH_NUM_SAMPLES="${HIGH_NUM_SAMPLES_CSV}"
export HIGH_N_STEPS="${HIGH_N_STEPS_CSV}"
export HIGH_TOPK="${HIGH_TOPK_CSV}"
export LOW_NUM_SAMPLES="${LOW_NUM_SAMPLES_CSV}"
export LOW_N_STEPS="${LOW_N_STEPS_CSV}"
export LOW_TOPK="${LOW_TOPK_CSV}"
export EVAL_SEEDS="${EVAL_SEEDS:-42}"
export EVAL_SEED="${EVAL_SEED_VALUE}"
export EVAL_DETERMINISM="${EVAL_DETERMINISM:-strict}"

export EVAL_SUBDIR="eval_hope_hierarchical_matrix_${DISPLAY_LABEL}_job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-${TASK_ID}}"
export RESULT_FILENAME="${MODEL_LABEL}_${DISPLAY_LABEL}_results.txt"

echo "Checkpoint row: $((CHECKPOINT_INDEX + 1)) / ${NUM_CHECKPOINTS}"
echo "Config row: ${CONFIG_INDEX} / ${NUM_CONFIGS}"
echo "Task: ${TASK_ID} / ${TOTAL_TASKS}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint epoch: ${CHECKPOINT_EPOCH}"
echo "Config note: ${CONFIG_NOTE:-<blank>}"
echo "Planning mode: ${PLANNING_MODE}"
echo "Goal offset tag: ${GOAL_OFFSET_TAG}"
echo "Goal offset steps: ${GOAL_OFFSET_STEPS}"
echo "Eval budget: ${EVAL_BUDGET}"
echo "Num eval: ${NUM_EVAL}"
echo "Eval device: ${EVAL_DEVICE}"
echo "Label: ${LABEL}"
echo "Display label: ${DISPLAY_LABEL}"
echo "Seed: ${EVAL_SEED}"
echo "Seeds in sweep: ${EVAL_SEEDS}"
echo "Determinism: ${EVAL_DETERMINISM}"
echo "High planner: horizon=${HIGH_HORIZON}, receding=${HIGH_RECEDING_HORIZON}, block=${HIGH_ACTION_BLOCK}, samples=${HIGH_NUM_SAMPLES}, iters=${HIGH_N_STEPS}, topk=${HIGH_TOPK}, replan=${HIGH_REPLAN_INTERVAL}"
echo "Low planner: horizon=${LOW_HORIZON}, receding=${LOW_RECEDING_HORIZON}, block=${LOW_ACTION_BLOCK}, samples=${LOW_NUM_SAMPLES}, iters=${LOW_N_STEPS}, topk=${LOW_TOPK}"
echo "Sweep file: ${SWEEP_FILE}"
echo "Checkpoint file: ${CHECKPOINT_FILE}"
echo "Base script: ${BASE_SCRIPT}"

exec bash "${BASE_SCRIPT}"
