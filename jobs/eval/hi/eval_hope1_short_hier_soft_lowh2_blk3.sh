#!/bin/bash

# Wrapper job: short hierarchical-soft eval with the current best low horizon,
# using the checkpoint-compatible low-level action block size.
#
# Suggested use:
#   cd jobs/eval/hi
#   sbatch eval_hope1_short_hier_soft_lowh2_blk3.sh

#SBATCH --partition=rome
#SBATCH --job-name=hi_eval_hope1_short_hier_soft_lowh2_blk5_compat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=eval_hope1_short_hier_soft_lowh2_blk5_compat_%j.out
#SBATCH --error=eval_hope1_short_hier_soft_lowh2_blk5_compat_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
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

export EVAL_SUBDIR="${EVAL_SUBDIR:-eval_hier_soft_lowh2_blk5_compat_d${GOAL_OFFSET_STEPS}_b${EVAL_BUDGET}_job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"

exec "${SCRIPT_DIR}/eval_hope1_short_hier_soft_lowerh.sh"
