#!/bin/bash

# Snellius job: original LeWM PushT eval with per-episode metrics and a larger
# evaluation budget than the baseline script.
#
# Relative to pusht_eval_withmetrics.sh:
# - eval.eval_budget: 50 -> 100
#
# Usage:
#   cd jobs/eval/original
#   sbatch pusht_eval_withmetrics_budget.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=orig_eval_pusht_budget
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=out/pusht_eval_withmetrics_budget_%j.out
#SBATCH --error=out/pusht_eval_withmetrics_budget_%j.err

VARIANT_NAME="${VARIANT_NAME:-budget100}"
EVAL_BUDGET="${EVAL_BUDGET:-100}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/pusht_eval_withmetrics.sh"
