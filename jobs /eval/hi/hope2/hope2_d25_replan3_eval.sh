#!/bin/bash

# Hope2 PushT eval on d=25 with the best prior short-horizon setup, but with
# more frequent high-level replanning.

#SBATCH --partition=rome
#SBATCH --job-name=hi_eval_hope2_d25_replan3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=hope2_d25_replan3_eval_%j.out
#SBATCH --error=hope2_d25_replan3_eval_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/hope2_pusht_eval_base.sh"
if [[ ! -f "${BASE_SCRIPT}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_SCRIPT="${SLURM_SUBMIT_DIR}/hope2_pusht_eval_base.sh"
fi
if [[ ! -f "${BASE_SCRIPT}" && -n "${PROJECT_ROOT:-}" ]]; then
  BASE_SCRIPT="${PROJECT_ROOT}/jobs/eval/hi/hope2/hope2_pusht_eval_base.sh"
fi
if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: Could not locate hope2_pusht_eval_base.sh" >&2
  echo "Checked runtime dir (${SCRIPT_DIR}), SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-<unset>}), and PROJECT_ROOT (${PROJECT_ROOT:-<unset>})." >&2
  exit 2
fi

export EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
export GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
export EVAL_BUDGET="${EVAL_BUDGET:-50}"

export HIGH_NUM_SAMPLES="${HIGH_NUM_SAMPLES:-900}"
export HIGH_N_STEPS="${HIGH_N_STEPS:-20}"
export HIGH_TOPK="${HIGH_TOPK:-10}"
export HIGH_HORIZON="${HIGH_HORIZON:-1}"
export HIGH_RECEDING_HORIZON="${HIGH_RECEDING_HORIZON:-1}"
export HIGH_ACTION_BLOCK="${HIGH_ACTION_BLOCK:-1}"
export HIGH_REPLAN_INTERVAL="${HIGH_REPLAN_INTERVAL:-3}"

export LOW_NUM_SAMPLES="${LOW_NUM_SAMPLES:-300}"
export LOW_N_STEPS="${LOW_N_STEPS:-30}"
export LOW_TOPK="${LOW_TOPK:-150}"
export LOW_HORIZON="${LOW_HORIZON:-2}"
export LOW_RECEDING_HORIZON="${LOW_RECEDING_HORIZON:-1}"
export LOW_ACTION_BLOCK="${LOW_ACTION_BLOCK:-5}"

export EVAL_SUBDIR="${EVAL_SUBDIR:-eval_hope2_d25_replan3_b${EVAL_BUDGET}_job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"

exec bash "${BASE_SCRIPT}"
