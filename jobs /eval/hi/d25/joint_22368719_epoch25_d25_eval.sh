#!/bin/bash

# Dedicated PushT eval for the joint end-to-end run hi_lewm_joint_22368719 at
# epoch 25, using the strongest short-horizon hierarchical settings we have for
# this checkpoint family.
#
# Suggested use:
#   cd jobs/eval/hi/d25
#   sbatch joint_22368719_epoch25_d25_eval.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=hi_eval_joint22368719_d25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=joint_22368719_epoch25_d25_eval_%j.out
#SBATCH --error=joint_22368719_epoch25_d25_eval_%j.err

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
      if [[ -f "${repo_root}/hi_eval.py" && -f "${repo_root}/config/eval/hi_pusht.yaml" ]]; then
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

BASE_SCRIPT="${REPO_ROOT}/jobs/eval/hi/d25/d25_hierarchical_soft_low_horizon_base_eval.sh"
if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: Could not locate d25_hierarchical_soft_low_horizon_base_eval.sh under ${REPO_ROOT}" >&2
  exit 2
fi

export RUN_NAME="${RUN_NAME:-hi_lewm_joint_22368719}"
export CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-25}"
export EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
export GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
export EVAL_BUDGET="${EVAL_BUDGET:-50}"

export HIGH_NUM_SAMPLES="${HIGH_NUM_SAMPLES:-900}"
export HIGH_N_STEPS="${HIGH_N_STEPS:-20}"
export HIGH_TOPK="${HIGH_TOPK:-10}"
export HIGH_HORIZON="${HIGH_HORIZON:-1}"
export HIGH_RECEDING_HORIZON="${HIGH_RECEDING_HORIZON:-1}"
export HIGH_ACTION_BLOCK="${HIGH_ACTION_BLOCK:-1}"
export HIGH_REPLAN_INTERVAL="${HIGH_REPLAN_INTERVAL:-5}"

export LOW_NUM_SAMPLES="${LOW_NUM_SAMPLES:-300}"
export LOW_N_STEPS="${LOW_N_STEPS:-30}"
export LOW_TOPK="${LOW_TOPK:-150}"
export LOW_HORIZON="${LOW_HORIZON:-2}"
export LOW_RECEDING_HORIZON="${LOW_RECEDING_HORIZON:-1}"
export LOW_ACTION_BLOCK="${LOW_ACTION_BLOCK:-5}"

export EVAL_SUBDIR="${EVAL_SUBDIR:-eval_joint_22368719_epoch25_d25_h1_lowh2_b${EVAL_BUDGET}_job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"

exec bash "${BASE_SCRIPT}"
