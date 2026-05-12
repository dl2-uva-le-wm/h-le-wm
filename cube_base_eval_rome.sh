#!/bin/bash
#
# CPU Slurm launcher for evaluating the base LeWorldModel on OGB Cube.
# Writes the aggregate success rate and a per-episode manifest under
# $STABLEWM_HOME/cube/${EVAL_SUBDIR}/.
#
# Usage:
#   sbatch cube_base_eval_rome.sh
# Optional overrides:
#   sbatch --export=ALL,STABLEWM_HOME=/scratch-shared/$USER/stablewm_data,EVAL_BUDGET=50 cube_base_eval_rome.sh

#SBATCH --partition=rome
#SBATCH --job-name=base_cube_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=cube_base_eval_%j.out
#SBATCH --error=cube_base_eval_%j.err

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

POLICY="${POLICY:-cube/lewm}"
CONFIG_NAME="${CONFIG_NAME:-cube}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
PLAN_HORIZON="${PLAN_HORIZON:-5}"
SOLVER_DEVICE="${SOLVER_DEVICE:-cpu}"
JOB_TOKEN="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval_cube_base_rome_${JOB_TOKEN}}"
RESULT_FILENAME="${RESULT_FILENAME:-ogb_cube_results_base.txt}"

CKPT_OBJECT_PATH="${STABLEWM_HOME}/${POLICY}_object.ckpt"
DATASET_PATH="${STABLEWM_HOME}/ogbench/cube_single_expert.h5"

if [[ ! -f "${CKPT_OBJECT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_OBJECT_PATH}" >&2
  exit 3
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: dataset not found: ${DATASET_PATH}" >&2
  exit 4
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
echo "POLICY=${POLICY}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "SOLVER_DEVICE=${SOLVER_DEVICE}"
echo "EVAL_BUDGET=${EVAL_BUDGET}"
echo "GOAL_OFFSET_STEPS=${GOAL_OFFSET_STEPS}"
echo "PLAN_HORIZON=${PLAN_HORIZON}"
echo "EVAL_SUBDIR=${EVAL_SUBDIR}"
echo "RESULT_FILENAME=${RESULT_FILENAME}"
echo "Checkpoint=${CKPT_OBJECT_PATH}"
echo "Dataset=${DATASET_PATH}"
echo ""
echo "Launching:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

RESULT_PATH="${STABLEWM_HOME}/cube/${EVAL_SUBDIR}/${RESULT_FILENAME}"
MANIFEST_PATH="${STABLEWM_HOME}/cube/${EVAL_SUBDIR}/${RESULT_FILENAME%.*}_episodes.tsv"

echo ""
echo "Eval finished."
echo "Results: ${RESULT_PATH}"
echo "Manifest: ${MANIFEST_PATH}"
echo ""
echo "Success-rate lines:"
rg -n "metrics:|success_rate" "${RESULT_PATH}" || true
