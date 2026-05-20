#!/bin/bash
#
# CPU Slurm array launcher for evaluating the official base LeWorldModel
# checkpoints on Cube and Reacher at d in {25, 50, 75}. This always stays on
# the original LeWM path via original_eval_with_manifest.py and does not use
# the hierarchical hi_eval.py implementation.
#
# By default this script is locked to the official baseline checkpoints:
#   - cube/lewm
#   - reacher/lewm
# It ignores the old POLICY/REACHER_POLICY/CUBE_POLICY env-var pattern so the
# run cannot silently switch to a locally trained checkpoint.
#
# Array mapping:
#   1 -> cube, d=25
#   2 -> cube, d=50
#   3 -> cube, d=75
#   4 -> reacher, d=25
#   5 -> reacher, d=50
#   6 -> reacher, d=75
#
# Usage:
#   sbatch cube_reacher_base_eval_rome.sh
# Safe overrides:
#   sbatch --export=ALL,STABLEWM_HOME=/scratch-shared/$USER/stablewm_data,EVAL_BUDGET=50 cube_reacher_base_eval_rome.sh
# Intentional non-official override:
#   sbatch --export=ALL,ALLOW_NON_OFFICIAL_POLICY=1,POLICY_OVERRIDE=reacher/my_model cube_reacher_base_eval_rome.sh

#SBATCH --partition=rome
#SBATCH --job-name=cube_reacher_base_eval
#SBATCH --array=1-6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=salvatore.lo.sardo@student.uva.nl
#SBATCH --output=cube_reacher_base_eval_%A_%a.out
#SBATCH --error=cube_reacher_base_eval_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

resolve_repo_root() {
  local c p level
  for c in \
    "${PROJECT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${SCRIPT_DIR}"; do
    [[ -z "${c}" ]] && continue
    p="${c}"
    for level in 0 1 2 3 4 5 6; do
      if p="$(cd "${p}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${p}/original_eval_with_manifest.py" && -d "${p}/third_party/lewm" ]]; then
          echo "${p}"
          return 0
        fi
      fi
      p="${p}/.."
    done
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root." >&2
  echo "Submit from inside the repo or pass PROJECT_ROOT=/path/to/h-lewm." >&2
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
  echo "Run jobs/setup/setup_env.sh first, or create the environment from environment-gpu.yml" >&2
  exit 2
fi
set -u

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

OFFICIAL_CUBE_POLICY="cube/lewm"
OFFICIAL_REACHER_POLICY="reacher/lewm"
ALLOW_NON_OFFICIAL_POLICY="${ALLOW_NON_OFFICIAL_POLICY:-0}"

ensure_reacher_dataset_layout() {
  local legacy_path canonical_path canonical_dir
  legacy_path="${STABLEWM_HOME}/reacher.h5"
  canonical_path="${STABLEWM_HOME}/dmc/reacher_random.h5"
  canonical_dir="$(dirname "${canonical_path}")"

  if [[ -f "${canonical_path}" || -L "${canonical_path}" ]]; then
    return 0
  fi

  if [[ -f "${legacy_path}" ]]; then
    mkdir -p "${canonical_dir}"
    ln -s "../reacher.h5" "${canonical_path}"
    echo "Normalized legacy reacher dataset path:"
    echo "  ${canonical_path} -> ${legacy_path}"
  fi
}

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set." >&2
  echo "Submit with sbatch cube_reacher_base_eval_rome.sh or pass TASK_ID=1..3." >&2
  exit 3
fi

case "${TASK_ID}" in
  1)
    ENV_NAME="cube"
    GOAL_OFFSET_STEPS=25
    ;;
  2)
    ENV_NAME="cube"
    GOAL_OFFSET_STEPS=50
    ;;
  3)
    ENV_NAME="cube"
    GOAL_OFFSET_STEPS=75
    ;;
  4)
    ENV_NAME="reacher"
    GOAL_OFFSET_STEPS=25
    ;;
  5)
    ENV_NAME="reacher"
    GOAL_OFFSET_STEPS=50
    ;;
  6)
    ENV_NAME="reacher"
    GOAL_OFFSET_STEPS=75
    ;;
  *)
    echo "ERROR: TASK_ID must be in 1..6, got '${TASK_ID}'." >&2
    exit 4
    ;;
esac

SOLVER_DEVICE="${SOLVER_DEVICE:-cpu}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
PLAN_HORIZON="${PLAN_HORIZON:-5}"
JOB_TOKEN="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"

case "${ENV_NAME}" in
  cube)
    OFFICIAL_POLICY="${OFFICIAL_CUBE_POLICY}"
    BASE_CONFIG_NAME="cube"
    DATASET_PATH="${STABLEWM_HOME}/ogbench/cube_single_expert.h5"
    RESULT_FILENAME="${CUBE_RESULT_FILENAME:-ogb_cube_results_base_d${GOAL_OFFSET_STEPS}.txt}"
    ;;
  reacher)
    OFFICIAL_POLICY="${OFFICIAL_REACHER_POLICY}"
    BASE_CONFIG_NAME="reacher"
    ensure_reacher_dataset_layout
    DATASET_PATH="${STABLEWM_HOME}/dmc/reacher_random.h5"
    RESULT_FILENAME="${REACHER_RESULT_FILENAME:-dmc_reacher_results_base_d${GOAL_OFFSET_STEPS}.txt}"
    ;;
  *)
    echo "ERROR: unsupported ENV_NAME '${ENV_NAME}'." >&2
    exit 4
    ;;
esac

if [[ "${ALLOW_NON_OFFICIAL_POLICY}" == "1" ]]; then
  POLICY="${POLICY_OVERRIDE:-${OFFICIAL_POLICY}}"
  CONFIG_NAME="${CONFIG_NAME_OVERRIDE:-${BASE_CONFIG_NAME}}"
else
  POLICY="${OFFICIAL_POLICY}"
  CONFIG_NAME="${BASE_CONFIG_NAME}"
fi

EVAL_SUBDIR="${EVAL_SUBDIR:-eval_${ENV_NAME}_base_d${GOAL_OFFSET_STEPS}_job_${JOB_TOKEN}_task_${TASK_ID}}"
CKPT_OBJECT_PATH="${STABLEWM_HOME}/${POLICY}_object.ckpt"
POLICY_PARENT_REL="$(dirname "${POLICY}")"
if [[ "${POLICY_PARENT_REL}" == "." ]]; then
  RESULT_BASE_DIR="${STABLEWM_HOME}"
else
  RESULT_BASE_DIR="${STABLEWM_HOME}/${POLICY_PARENT_REL}"
fi
RESULT_PATH="${RESULT_BASE_DIR}/${EVAL_SUBDIR}/${RESULT_FILENAME}"
MANIFEST_PATH="${RESULT_PATH%.*}_episodes.tsv"

if [[ ! -f "${CKPT_OBJECT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_OBJECT_PATH}" >&2
  if [[ "${ENV_NAME}" == "cube" ]]; then
    echo "Run jobs/setup/download_cube_reacher_assets.sh first." >&2
  else
    echo "Run jobs/setup/download_reacher_model.sh first." >&2
  fi
  exit 5
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: dataset not found: ${DATASET_PATH}" >&2
  if [[ "${ENV_NAME}" == "cube" ]]; then
    echo "Run jobs/setup/download_cube_reacher_assets.sh first." >&2
  else
    echo "Run jobs/setup/download_reacher.sh first." >&2
  fi
  exit 6
fi

cd "${REPO_ROOT}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"
fi

CMD=(
  python original_eval_with_manifest.py
  --config-name="${CONFIG_NAME}"
  "policy=${POLICY}"
  "solver.device=${SOLVER_DEVICE}"
  "eval.eval_budget=${EVAL_BUDGET}"
  "eval.goal_offset_steps=${GOAL_OFFSET_STEPS}"
  "plan_config.horizon=${PLAN_HORIZON}"
  "output.filename=${RESULT_FILENAME}"
  "+output.subdir=${EVAL_SUBDIR}"
)

echo "REPO_ROOT=${REPO_ROOT}"
echo "STABLEWM_HOME=${STABLEWM_HOME}"
echo "TASK_ID=${TASK_ID}"
echo "ENV_NAME=${ENV_NAME}"
echo "OFFICIAL_POLICY=${OFFICIAL_POLICY}"
echo "POLICY=${POLICY}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "SOLVER_DEVICE=${SOLVER_DEVICE}"
echo "EVAL_BUDGET=${EVAL_BUDGET}"
echo "GOAL_OFFSET_STEPS=${GOAL_OFFSET_STEPS}"
echo "PLAN_HORIZON=${PLAN_HORIZON}"
echo "ALLOW_NON_OFFICIAL_POLICY=${ALLOW_NON_OFFICIAL_POLICY}"
echo "EVAL_SUBDIR=${EVAL_SUBDIR}"
echo "RESULT_FILENAME=${RESULT_FILENAME}"
echo "Checkpoint=${CKPT_OBJECT_PATH}"
echo "Dataset=${DATASET_PATH}"
echo ""
echo "Launching:"
printf '  %q' "${CMD[@]}"
echo
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, not launching eval."
  exit 0
fi

"${CMD[@]}"

echo ""
echo "Eval finished."
echo "Results: ${RESULT_PATH}"
echo "Manifest: ${MANIFEST_PATH}"
echo ""
echo "Success-rate lines:"
rg -n "metrics:|success_rate" "${RESULT_PATH}" || true
