#!/bin/bash

# Lower-level health check for the joint end-to-end hierarchical PushT run.
#
# Purpose:
# - Load the hierarchical end-to-end object checkpoint
# - Bypass hierarchical planning entirely
# - Evaluate it with the original flat LeWM eval path and d=25 settings
# - Check whether the learned lower level still matches the flat baseline regime
#
# Why this exists:
# - `hi_eval.py planning.mode=flat` is not the baseline-equivalent control.
# - This script instead uses `original_eval_with_manifest.py`, which mirrors the
#   original LeWM evaluation path.
#
# Default checkpoint:
# - RUN_NAME=hi_lewm_joint_22368719
# - CHECKPOINT_EPOCH=25
#
# Usage:
#   cd jobs/eval/hi/d25
#   sbatch epoch_25_healthcheck_e2e_lower_level.sh
#
# Optional overrides:
#   sbatch --export=ALL,CHECKPOINT_EPOCH=latest epoch_25_healthcheck_e2e_lower_level.sh
#   sbatch --export=ALL,CHECKPOINT_EPOCH=20 epoch_25_healthcheck_e2e_lower_level.sh
#   sbatch --export=ALL,RUN_NAME=hi_lewm_joint_22368719 epoch_25_healthcheck_e2e_lower_level.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=e2e_ll_health_ep25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=epoch_25_healthcheck_e2e_lower_level_%j.out
#SBATCH --error=epoch_25_healthcheck_e2e_lower_level_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

resolve_repo_root() {
  local candidate repo_root
  for candidate in \
    "${PROJECT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${SCRIPT_DIR:-}" \
    "${HOME}/main" \
    "/gpfs/home2/${USER}/main" \
    "${HOME}/h-le-wm" \
    "${HOME}/h-lewm" \
    "/gpfs/home2/${USER}/h-le-wm" \
    "/gpfs/home2/${USER}/h-lewm"; do
    [[ -z "${candidate}" ]] && continue
    if ! repo_root="$(cd "${candidate}" >/dev/null 2>&1 && pwd)"; then
      continue
    fi

    while :; do
      if [[ -f "${repo_root}/original_eval_with_manifest.py" && -f "${repo_root}/third_party/lewm/config/eval/pusht.yaml" ]]; then
        echo "${repo_root}"
        return 0
      fi
      [[ "${repo_root}" == "/" ]] && break
      repo_root="$(dirname "${repo_root}")"
    done
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root." >&2
  echo "Submit from repo root or pass PROJECT_ROOT=/path/to/h-le-wm" >&2
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
RUN_NAME="${RUN_NAME:-hi_lewm_joint_22368719}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-25}"  # "latest" or integer >= 1
CONFIG_NAME="${CONFIG_NAME:-pusht.yaml}"

# Explicitly pin the original flat d25 LeWM eval recipe.
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
PLAN_HORIZON="${PLAN_HORIZON:-5}"
PLAN_RECEDING_HORIZON="${PLAN_RECEDING_HORIZON:-5}"
PLAN_ACTION_BLOCK="${PLAN_ACTION_BLOCK:-5}"
SOLVER_NUM_SAMPLES="${SOLVER_NUM_SAMPLES:-300}"
SOLVER_N_STEPS="${SOLVER_N_STEPS:-30}"
SOLVER_TOPK="${SOLVER_TOPK:-30}"
SOLVER_VAR_SCALE="${SOLVER_VAR_SCALE:-1.0}"

RUN_DIR="${STABLEWM_HOME}/runs/${RUN_NAME}"
DATASET_PATH="${STABLEWM_HOME}/pusht_expert_train.h5"

if [[ ! -f "${REPO_ROOT}/original_eval_with_manifest.py" ]]; then
  echo "ERROR: original_eval_with_manifest.py not found at ${REPO_ROOT}" >&2
  exit 3
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: dataset file not found: ${DATASET_PATH}" >&2
  echo "Run setup first, for example:" >&2
  echo "  sbatch --export=ALL,STABLEWM_HOME=${STABLEWM_HOME} jobs/setup/download_pusht.sh" >&2
  exit 4
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run directory not found: ${RUN_DIR}" >&2
  exit 5
fi

if [[ "${CHECKPOINT_EPOCH}" == "latest" ]]; then
  shopt -s nullglob
  candidates=( "${RUN_DIR}/${RUN_NAME}"_epoch_*_object.ckpt )
  shopt -u nullglob
  if (( ${#candidates[@]} == 0 )); then
    echo "ERROR: no object checkpoints found in ${RUN_DIR}" >&2
    echo "Expected pattern: ${RUN_NAME}_epoch_*_object.ckpt" >&2
    exit 6
  fi
  mapfile -t sorted_candidates < <(printf '%s\n' "${candidates[@]}" | sort -V)
  CKPT_OBJECT_PATH="${sorted_candidates[${#sorted_candidates[@]}-1]}"
else
  if ! [[ "${CHECKPOINT_EPOCH}" =~ ^[0-9]+$ ]] || (( CHECKPOINT_EPOCH < 1 )); then
    echo "ERROR: CHECKPOINT_EPOCH must be 'latest' or an integer >= 1, got '${CHECKPOINT_EPOCH}'" >&2
    exit 7
  fi
  CKPT_OBJECT_PATH="${RUN_DIR}/${RUN_NAME}_epoch_${CHECKPOINT_EPOCH}_object.ckpt"
fi

if [[ ! -f "${CKPT_OBJECT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_OBJECT_PATH}" >&2
  echo "Available checkpoints in ${RUN_DIR}:" >&2
  ls -1 "${RUN_DIR}"/*_object.ckpt >&2 || true
  exit 8
fi

if [[ "${CKPT_OBJECT_PATH}" != "${STABLEWM_HOME}/"* ]]; then
  echo "ERROR: checkpoint is not under STABLEWM_HOME; cannot derive policy path." >&2
  echo "STABLEWM_HOME=${STABLEWM_HOME}" >&2
  echo "CKPT_OBJECT_PATH=${CKPT_OBJECT_PATH}" >&2
  exit 9
fi

POLICY="${CKPT_OBJECT_PATH#${STABLEWM_HOME}/}"
POLICY="${POLICY%_object.ckpt}"
POLICY_BASENAME="$(basename "${POLICY}")"
VARIANT_NAME="${VARIANT_NAME:-e2e_lower_level_healthcheck_d25}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval_${VARIANT_NAME}_${POLICY_BASENAME}_job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
RESULT_FILENAME="${RESULT_FILENAME:-${POLICY_BASENAME}_${VARIANT_NAME}.txt}"
ARTIFACTS_DIR="${STABLEWM_HOME}/$(dirname "${POLICY}")/${EVAL_SUBDIR}"

echo "Repo root: ${REPO_ROOT}"
echo "Script dir: ${SCRIPT_DIR}"
echo "STABLEWM_HOME: ${STABLEWM_HOME}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint selection: ${CHECKPOINT_EPOCH}"
echo "Checkpoint object: ${CKPT_OBJECT_PATH}"
echo "Policy arg for original eval: ${POLICY}"
echo "Config name: ${CONFIG_NAME}"
echo "Eval mode: original flat LeWM path"
echo "Goal offset steps (d): ${GOAL_OFFSET_STEPS}"
echo "Eval budget: ${EVAL_BUDGET}"
echo "Flat plan horizon: ${PLAN_HORIZON}"
echo "Flat receding horizon: ${PLAN_RECEDING_HORIZON}"
echo "Flat action block: ${PLAN_ACTION_BLOCK}"
echo "CEM num_samples: ${SOLVER_NUM_SAMPLES}"
echo "CEM n_steps: ${SOLVER_N_STEPS}"
echo "CEM topk: ${SOLVER_TOPK}"
echo "CEM var_scale: ${SOLVER_VAR_SCALE}"
echo "Output subdir: ${EVAL_SUBDIR}"
echo "Artifacts dir: ${ARTIFACTS_DIR}"
echo "Result file: ${ARTIFACTS_DIR}/${RESULT_FILENAME}"
echo "Health-check intent: evaluate only the lower-level / flat control path."

cd "${REPO_ROOT}"

# Compatibility for object checkpoints pickled from baseline code:
# torch.load may need top-level imports like `module` / `utils` from third_party/lewm.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"
fi
echo "PYTHONPATH prefix: ${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"

CMD=(
  python original_eval_with_manifest.py
  --config-name="${CONFIG_NAME}"
  "policy=${POLICY}"
  "eval.goal_offset_steps=${GOAL_OFFSET_STEPS}"
  "eval.eval_budget=${EVAL_BUDGET}"
  "plan_config.horizon=${PLAN_HORIZON}"
  "plan_config.receding_horizon=${PLAN_RECEDING_HORIZON}"
  "plan_config.action_block=${PLAN_ACTION_BLOCK}"
  "solver.num_samples=${SOLVER_NUM_SAMPLES}"
  "solver.n_steps=${SOLVER_N_STEPS}"
  "solver.topk=${SOLVER_TOPK}"
  "solver.var_scale=${SOLVER_VAR_SCALE}"
  "output.filename=${RESULT_FILENAME}"
  "+output.subdir=${EVAL_SUBDIR}"
)

echo ""
echo "==> Launching lower-level health-check eval:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo ""
echo "Eval finished."
echo "Artifacts written to: ${ARTIFACTS_DIR}"
echo "Episode manifest should be under: ${ARTIFACTS_DIR}/${RESULT_FILENAME%.*}_episodes.tsv"
