#!/bin/bash

# PushT eval for the timed-out hope3 latent-dim-8 run, pinned to its last
# saved object checkpoint (epoch 13), using the standard d=25 H=2 GPU setup.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=hi_eval_h3e13_d25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=hope3_ep13_d25_eval_%j.out
#SBATCH --error=hope3_ep13_d25_eval_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/../hope2/hope2_pusht_eval_base.sh"
if [[ ! -f "${BASE_SCRIPT}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_SCRIPT="${SLURM_SUBMIT_DIR}/../hope2/hope2_pusht_eval_base.sh"
fi
if [[ ! -f "${BASE_SCRIPT}" && -n "${PROJECT_ROOT:-}" ]]; then
  BASE_SCRIPT="${PROJECT_ROOT}/jobs/eval/hi/hope2/hope2_pusht_eval_base.sh"
fi
if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: Could not locate hope2_pusht_eval_base.sh" >&2
  exit 2
fi

export RUN_NAME="${RUN_NAME:-hi_lewm_p2_train_hope3_22515001}"
export CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-13}"
export MODEL_LABEL="${MODEL_LABEL:-hope3_ep13}"

export EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
export GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
export EVAL_BUDGET="${EVAL_BUDGET:-50}"

export HIGH_NUM_SAMPLES="${HIGH_NUM_SAMPLES:-900}"
export HIGH_N_STEPS="${HIGH_N_STEPS:-20}"
export HIGH_TOPK="${HIGH_TOPK:-10}"
export HIGH_HORIZON="${HIGH_HORIZON:-2}"
export HIGH_RECEDING_HORIZON="${HIGH_RECEDING_HORIZON:-1}"
export HIGH_ACTION_BLOCK="${HIGH_ACTION_BLOCK:-1}"
export HIGH_REPLAN_INTERVAL="${HIGH_REPLAN_INTERVAL:-5}"

export LOW_NUM_SAMPLES="${LOW_NUM_SAMPLES:-300}"
export LOW_N_STEPS="${LOW_N_STEPS:-30}"
export LOW_TOPK="${LOW_TOPK:-150}"
export LOW_HORIZON="${LOW_HORIZON:-2}"
export LOW_RECEDING_HORIZON="${LOW_RECEDING_HORIZON:-1}"
export LOW_ACTION_BLOCK="${LOW_ACTION_BLOCK:-5}"

export EVAL_SUBDIR="${EVAL_SUBDIR:-eval_${MODEL_LABEL}_d25_h2_b${EVAL_BUDGET}_job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export RESULT_FILENAME="${RESULT_FILENAME:-${MODEL_LABEL}_d25_h2_results.txt}"

exec bash "${BASE_SCRIPT}"
