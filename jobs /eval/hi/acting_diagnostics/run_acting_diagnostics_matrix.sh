#!/bin/bash

# Array-driven acting diagnostics matrix for hierarchical PushT checkpoints.

#SBATCH --partition=rome
#SBATCH --gpus=0
#SBATCH --job-name=hi_act_diag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --chdir=/gpfs/home2/scur0200/main/jobs/eval/hi/acting_diagnostics
#SBATCH --output=run_acting_diagnostics_matrix_%A_%a.out
#SBATCH --error=run_acting_diagnostics_matrix_%A_%a.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_INTERACTIVE:-0}" != "1" ]]; then
  echo "ERROR: run_acting_diagnostics_matrix.sh is the worker script." >&2
  echo "Run it via Slurm with ./submit_acting_diagnostics_matrix.sh or sbatch." >&2
  echo "If you intentionally want a login-node debug run, set ALLOW_INTERACTIVE=1." >&2
  exit 10
fi

resolve_repo_root() {
  local c p
  for c in \
    "${PROJECT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${HOME}/main" \
    "/gpfs/home2/${USER}/main"; do
    [[ -z "${c}" ]] && continue
    for p in "${c}" "${c}/.." "${c}/../.." "${c}/../../.." "${c}/../../../.."; do
      if p="$(cd "${p}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${p}/scripts/run_hi_acting_diagnostic.py" && -f "${p}/hi_eval.py" ]]; then
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

module purge
module load 2025
module load Anaconda3/2025.06-1

set +u
eval "$(conda shell.bash hook)"
if conda env list | grep -E '(^|[[:space:]])lewm-gpu([[:space:]]|$)' >/dev/null 2>&1; then
  conda activate lewm-gpu
elif conda env list | grep -E '(^|[[:space:]])lewm([[:space:]]|$)' >/dev/null 2>&1; then
  conda activate lewm
else
  echo "ERROR: Could not find conda environment 'lewm-gpu' or 'lewm'" >&2
  exit 2
fi
set -u

SCRIPT_DIR="${JOB_SCRIPT_DIR:-}"
if [[ -z "${SCRIPT_DIR}" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
CHECKPOINT_FILE="${CHECKPOINT_FILE:-${SCRIPT_DIR}/checkpoints_acting.txt}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_ROOT}"

if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
  echo "ERROR: checkpoint list not found: ${CHECKPOINT_FILE}" >&2
  exit 3
fi

mapfile -t CHECKPOINT_ROWS < <(grep -Ev '^[[:space:]]*($|#)' "${CHECKPOINT_FILE}")
NUM_CHECKPOINTS="${#CHECKPOINT_ROWS[@]}"
NUM_CONFIGS=12

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set." >&2
  exit 4
fi

if [[ -n "${CHECKPOINT_ROW_INDEX:-}" ]]; then
  if ! [[ "${CHECKPOINT_ROW_INDEX}" =~ ^[0-9]+$ ]] || (( CHECKPOINT_ROW_INDEX < 1 || CHECKPOINT_ROW_INDEX > NUM_CHECKPOINTS )); then
    echo "ERROR: CHECKPOINT_ROW_INDEX ${CHECKPOINT_ROW_INDEX} is out of range 1-${NUM_CHECKPOINTS}" >&2
    exit 5
  fi
  if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 1 || TASK_ID > NUM_CONFIGS )); then
    echo "ERROR: task id ${TASK_ID} is out of range 1-${NUM_CONFIGS}" >&2
    exit 6
  fi
  CHECKPOINT_INDEX=$(( CHECKPOINT_ROW_INDEX - 1 ))
  CONFIG_INDEX="${TASK_ID}"
else
  if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 1 || TASK_ID > NUM_CHECKPOINTS * NUM_CONFIGS )); then
    echo "ERROR: task id ${TASK_ID} is out of range 1-$((NUM_CHECKPOINTS * NUM_CONFIGS))" >&2
    exit 6
  fi
  CHECKPOINT_INDEX=$(( (TASK_ID - 1) / NUM_CONFIGS ))
  CONFIG_INDEX=$(( (TASK_ID - 1) % NUM_CONFIGS + 1 ))
fi

read -r RUN_NAME CHECKPOINT_EPOCH <<< "${CHECKPOINT_ROWS[CHECKPOINT_INDEX]}"
if [[ -z "${RUN_NAME:-}" || -z "${CHECKPOINT_EPOCH:-}" ]]; then
  echo "ERROR: invalid checkpoint row: ${CHECKPOINT_ROWS[CHECKPOINT_INDEX]}" >&2
  exit 7
fi

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
RUN_DIR="${STABLEWM_HOME}/runs/${RUN_NAME}"
CKPT_OBJECT_PATH="${RUN_DIR}/${RUN_NAME}_epoch_${CHECKPOINT_EPOCH}_object.ckpt"
if [[ ! -f "${CKPT_OBJECT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_OBJECT_PATH}" >&2
  exit 8
fi

POLICY="${CKPT_OBJECT_PATH#${STABLEWM_HOME}/}"
POLICY="${POLICY%_object.ckpt}"

EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-50}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
FRAME_SKIP="${FRAME_SKIP:-5}"
NUM_EVAL="${NUM_EVAL:-50}"
NUM_REFERENCE_SAMPLES="${NUM_REFERENCE_SAMPLES:-4096}"
SUBGOAL_OFFSETS="${SUBGOAL_OFFSETS:-2,3,5}"
SEED="${SEED:-42}"

EXPERIMENT_KIND=""
HIGH_HORIZON=2
LOW_HORIZON=2
LOW_RECEDING_HORIZON=1
HIGH_NUM_SAMPLES="${HIGH_NUM_SAMPLES:-1500}"
HIGH_ITERS="${HIGH_ITERS:-40}"
HIGH_TOPK="${HIGH_TOPK:-10}"
LOW_NUM_SAMPLES="${LOW_NUM_SAMPLES:-900}"
LOW_ITERS="${LOW_ITERS:-20}"
LOW_TOPK="${LOW_TOPK:-150}"

case "${CONFIG_INDEX}" in
  1)  EXPERIMENT_KIND="oracle_subgoal_acting"; HIGH_HORIZON=2; LOW_HORIZON=2; LOW_RECEDING_HORIZON=1 ;;
  2)  EXPERIMENT_KIND="oracle_subgoal_acting"; HIGH_HORIZON=2; LOW_HORIZON=3; LOW_RECEDING_HORIZON=1 ;;
  3)  EXPERIMENT_KIND="oracle_subgoal_acting"; HIGH_HORIZON=2; LOW_HORIZON=5; LOW_RECEDING_HORIZON=1 ;;
  4)  EXPERIMENT_KIND="low_level_reality_gap"; HIGH_HORIZON=1; LOW_HORIZON=2; LOW_RECEDING_HORIZON=1 ;;
  5)  EXPERIMENT_KIND="low_level_reality_gap"; HIGH_HORIZON=1; LOW_HORIZON=3; LOW_RECEDING_HORIZON=1 ;;
  6)  EXPERIMENT_KIND="low_level_reality_gap"; HIGH_HORIZON=1; LOW_HORIZON=5; LOW_RECEDING_HORIZON=1 ;;
  7)  EXPERIMENT_KIND="generated_subgoal_acting"; HIGH_HORIZON=1; LOW_HORIZON=2; LOW_RECEDING_HORIZON=1 ;;
  8)  EXPERIMENT_KIND="generated_subgoal_acting"; HIGH_HORIZON=2; LOW_HORIZON=2; LOW_RECEDING_HORIZON=1 ;;
  9)  EXPERIMENT_KIND="online_hierarchical_logging"; HIGH_HORIZON=1; LOW_HORIZON=2; LOW_RECEDING_HORIZON=1 ;;
  10) EXPERIMENT_KIND="online_hierarchical_logging"; HIGH_HORIZON=2; LOW_HORIZON=2; LOW_RECEDING_HORIZON=1 ;;
  11) EXPERIMENT_KIND="online_hierarchical_logging"; HIGH_HORIZON=2; LOW_HORIZON=5; LOW_RECEDING_HORIZON=1 ;;
  12) EXPERIMENT_KIND="online_hierarchical_logging"; HIGH_HORIZON=2; LOW_HORIZON=5; LOW_RECEDING_HORIZON=5 ;;
  *) echo "ERROR: unsupported config index ${CONFIG_INDEX}" >&2; exit 9 ;;
esac

RUN_LABEL="${EXPERIMENT_KIND}_d${GOAL_OFFSET_STEPS}_hh${HIGH_HORIZON}_lh${LOW_HORIZON}_lrh${LOW_RECEDING_HORIZON}"
ARTIFACT_SUBDIR="acting_${RUN_LABEL}_job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-${TASK_ID}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${RUN_DIR}}"
ARTIFACT_DIR="${ARTIFACT_ROOT%/}/${ARTIFACT_SUBDIR}"
mkdir -p "${ARTIFACT_DIR}"
CACHE_DIR="${CACHE_DIR:-${STABLEWM_HOME}}"
mkdir -p "${CACHE_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ARTIFACT_DIR}/mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ARTIFACT_DIR}/xdg-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

JSON_PATH="${ARTIFACT_DIR}/${RUN_LABEL}.json"
NPZ_PATH="${ARTIFACT_DIR}/${RUN_LABEL}.npz"
TSV_PATH="${LOG_ROOT}/summary_${EXPERIMENT_KIND}.tsv"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"
fi

if [[ "${EVAL_DEVICE}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

echo "Repo root: ${REPO_ROOT}"
echo "Checkpoint row: $((CHECKPOINT_INDEX + 1)) / ${NUM_CHECKPOINTS}"
echo "Config row: ${CONFIG_INDEX} / ${NUM_CONFIGS}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint epoch: ${CHECKPOINT_EPOCH}"
echo "Policy: ${POLICY}"
echo "Experiment kind: ${EXPERIMENT_KIND}"
echo "Goal offset steps: ${GOAL_OFFSET_STEPS}"
echo "Eval budget: ${EVAL_BUDGET}"
echo "High horizon: ${HIGH_HORIZON}"
echo "Low horizon: ${LOW_HORIZON}"
echo "Low receding horizon: ${LOW_RECEDING_HORIZON}"
echo "High CEM: samples=${HIGH_NUM_SAMPLES}, iters=${HIGH_ITERS}, topk=${HIGH_TOPK}"
echo "Low CEM: samples=${LOW_NUM_SAMPLES}, iters=${LOW_ITERS}, topk=${LOW_TOPK}"
echo "Artifact root: ${ARTIFACT_ROOT}"
echo "Artifact dir: ${ARTIFACT_DIR}"
echo "Cache dir: ${CACHE_DIR}"
echo "JSON path: ${JSON_PATH}"
echo "NPZ path: ${NPZ_PATH}"
echo "TSV path: ${TSV_PATH}"

cd "${REPO_ROOT}"

CMD=(
  python scripts/run_hi_acting_diagnostic.py
  --policy "${POLICY}"
  --experiment-kind "${EXPERIMENT_KIND}"
  --eval-config "${REPO_ROOT}/config/eval/hi_pusht.yaml"
  --cache-dir "${CACHE_DIR}"
  --num-eval "${NUM_EVAL}"
  --goal-offset-steps "${GOAL_OFFSET_STEPS}"
  --eval-budget "${EVAL_BUDGET}"
  --high-horizon "${HIGH_HORIZON}"
  --low-horizon "${LOW_HORIZON}"
  --low-receding-horizon "${LOW_RECEDING_HORIZON}"
  --high-num-samples "${HIGH_NUM_SAMPLES}"
  --high-iters "${HIGH_ITERS}"
  --high-topk "${HIGH_TOPK}"
  --low-num-samples "${LOW_NUM_SAMPLES}"
  --low-iters "${LOW_ITERS}"
  --low-topk "${LOW_TOPK}"
  --frame-skip "${FRAME_SKIP}"
  --subgoal-offsets "${SUBGOAL_OFFSETS}"
  --num-reference-samples "${NUM_REFERENCE_SAMPLES}"
  --seed "${SEED}"
  --device "${EVAL_DEVICE}"
  --save-json "${JSON_PATH}"
  --save-npz "${NPZ_PATH}"
  --append-tsv "${TSV_PATH}"
)

echo ""
echo "==> Launching acting diagnostic command:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo ""
echo "Acting diagnostic finished."
echo "Artifacts written to: ${ARTIFACT_DIR}"
