#!/bin/bash

# Wrapper job: start from the best d25 hierarchical-soft low-h2 setup and
# adapt it to d50 with paper-inspired CEM scaling for samples/iterations.
#
# Design choices:
# - Keep low-level horizon at 2 (requested; strongest d25 setting).
# - Increase CEM samples/iterations toward HLWM d50-style budgets:
#   high-level: samples=1500, iters=40
#   low-level:  samples=900,  iters=20
#
# Suggested use:
#   cd jobs/eval/hi/d50
#   sbatch d50_hierarchical_soft_low_h2_paper_scaled_eval.sh

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=hi_eval_d50_hier_soft_low_h2_pscaled
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=d50_hierarchical_soft_low_h2_paper_scaled_eval_%j.out
#SBATCH --error=d50_hierarchical_soft_low_h2_paper_scaled_eval_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOWERH_SCRIPT="${SCRIPT_DIR}/../d25/d25_hierarchical_soft_low_horizon_base_eval.sh"
if [[ ! -f "${LOWERH_SCRIPT}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	LOWERH_SCRIPT="${SLURM_SUBMIT_DIR}/jobs/eval/hi/d25/d25_hierarchical_soft_low_horizon_base_eval.sh"
	if [[ ! -f "${LOWERH_SCRIPT}" ]]; then
		LOWERH_SCRIPT="${SLURM_SUBMIT_DIR}/../d25/d25_hierarchical_soft_low_horizon_base_eval.sh"
	fi
fi
if [[ ! -f "${LOWERH_SCRIPT}" && -n "${PROJECT_ROOT:-}" ]]; then
	LOWERH_SCRIPT="${PROJECT_ROOT}/jobs/eval/hi/d25/d25_hierarchical_soft_low_horizon_base_eval.sh"
fi
if [[ ! -f "${LOWERH_SCRIPT}" ]]; then
	echo "ERROR: Could not locate d25_hierarchical_soft_low_horizon_base_eval.sh" >&2
	echo "Checked runtime dir (${SCRIPT_DIR}), SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-<unset>}), and PROJECT_ROOT (${PROJECT_ROOT:-<unset>})." >&2
	exit 2
fi

export EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
export GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-50}"
export EVAL_BUDGET="${EVAL_BUDGET:-50}"

# Paper-inspired d50 scaling for CEM compute.
export HIGH_NUM_SAMPLES="${HIGH_NUM_SAMPLES:-1500}"
export HIGH_N_STEPS="${HIGH_N_STEPS:-40}"
export HIGH_TOPK="${HIGH_TOPK:-10}"
export HIGH_HORIZON="${HIGH_HORIZON:-1}"
export HIGH_RECEDING_HORIZON="${HIGH_RECEDING_HORIZON:-1}"
export HIGH_ACTION_BLOCK="${HIGH_ACTION_BLOCK:-1}"
export HIGH_REPLAN_INTERVAL="${HIGH_REPLAN_INTERVAL:-5}"

export LOW_NUM_SAMPLES="${LOW_NUM_SAMPLES:-900}"
export LOW_N_STEPS="${LOW_N_STEPS:-20}"
export LOW_TOPK="${LOW_TOPK:-150}"
export LOW_HORIZON="${LOW_HORIZON:-2}"
export LOW_RECEDING_HORIZON="${LOW_RECEDING_HORIZON:-1}"
export LOW_ACTION_BLOCK="${LOW_ACTION_BLOCK:-5}"

export EVAL_SUBDIR="${EVAL_SUBDIR:-eval_hier_soft_lowh2_d50_paperscaled_b${EVAL_BUDGET}_job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"

exec bash "${LOWERH_SCRIPT}"
