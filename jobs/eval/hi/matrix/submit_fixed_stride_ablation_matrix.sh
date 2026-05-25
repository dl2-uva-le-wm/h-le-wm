#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-${SCRIPT_DIR}/checkpoints_ablation_reducted.txt}"
SWEEP_FILE="${SWEEP_FILE:-${SCRIPT_DIR}/ablation_reducted_matrix_sweep.csv}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval_hope_hierarchical_matrix.sh}"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/run_hi_pusht_matrix_eval.sh}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs_hope_hierarchical_matrix}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu_h100}"
SBATCH_GPUS="${SBATCH_GPUS:-1}"

export CHECKPOINT_FILE
export SWEEP_FILE
export JOB_SCRIPT
export BASE_SCRIPT
export LOG_ROOT
export EVAL_DEVICE
export SBATCH_PARTITION
export SBATCH_GPUS

exec "${SCRIPT_DIR}/submit_hope_hierarchical_matrix.sh"
