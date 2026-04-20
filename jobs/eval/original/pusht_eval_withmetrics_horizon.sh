#!/bin/bash

# Snellius job: original LeWM PushT eval with per-episode metrics and a larger
# planning horizon than the baseline script.
#
# Relative to pusht_eval_withmetrics.sh:
# - plan_config.horizon: 5 -> 10
#
# Usage:
#   cd jobs/eval/original
#   sbatch pusht_eval_withmetrics_horizon.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=orig_eval_pusht_horizon
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=out/pusht_eval_withmetrics_horizon_%j.out
#SBATCH --error=out/pusht_eval_withmetrics_horizon_%j.err

VARIANT_NAME="${VARIANT_NAME:-horizon10}"
PLAN_HORIZON="${PLAN_HORIZON:-10}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/pusht_eval_withmetrics.sh"
