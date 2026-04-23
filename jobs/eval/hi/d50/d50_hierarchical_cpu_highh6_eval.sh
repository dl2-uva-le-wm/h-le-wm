#!/bin/bash

# Wrapper job: run d=50 hierarchical eval on CPU with high_horizon=6.
#
# Suggested use:
#   cd jobs/eval/hi/d50
#   sbatch d50_hierarchical_cpu_highh6_eval.sh
#
# Optional overrides:
#   sbatch --export=ALL,CHECKPOINT_EPOCH=10 d50_hierarchical_cpu_highh6_eval.sh
#   sbatch --export=ALL,HIGH_HORIZON=8 d50_hierarchical_cpu_highh6_eval.sh

#SBATCH --partition=rome
#SBATCH --gpus=0
#SBATCH --job-name=hi_eval_d50_cpu_highh6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=d50_hierarchical_cpu_highh6_eval_%j.out
#SBATCH --error=d50_hierarchical_cpu_highh6_eval_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/d50_hierarchical_soft_low_h2_paper_scaled_eval.sh"
if [[ ! -f "${BASE_SCRIPT}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_SCRIPT="${SLURM_SUBMIT_DIR}/jobs/eval/hi/d50/d50_hierarchical_soft_low_h2_paper_scaled_eval.sh"
  if [[ ! -f "${BASE_SCRIPT}" ]]; then
    BASE_SCRIPT="${SLURM_SUBMIT_DIR}/d50_hierarchical_soft_low_h2_paper_scaled_eval.sh"
  fi
fi
if [[ ! -f "${BASE_SCRIPT}" && -n "${PROJECT_ROOT:-}" ]]; then
  BASE_SCRIPT="${PROJECT_ROOT}/jobs/eval/hi/d50/d50_hierarchical_soft_low_h2_paper_scaled_eval.sh"
fi
if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: Could not locate d50_hierarchical_soft_low_h2_paper_scaled_eval.sh" >&2
  echo "Checked runtime dir (${SCRIPT_DIR}), SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-<unset>}), and PROJECT_ROOT (${PROJECT_ROOT:-<unset>})." >&2
  exit 2
fi

export EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
export GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-50}"
export EVAL_BUDGET="${EVAL_BUDGET:-50}"
export HIGH_HORIZON="${HIGH_HORIZON:-6}"
export EVAL_SUBDIR="${EVAL_SUBDIR:-eval_hier_soft_lowh2_d50_cpu_highh${HIGH_HORIZON}_b${EVAL_BUDGET}_job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"

exec bash "${BASE_SCRIPT}"
