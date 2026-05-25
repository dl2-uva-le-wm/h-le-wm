#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-${SCRIPT_DIR}/checkpoints_vq_ablation.txt}"
SWEEP_FILE="${SWEEP_FILE:-${SCRIPT_DIR}/ablation_reducted_matrix_sweep.csv}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval_hope_hierarchical_matrix.sh}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs_vq_ablation_matrix}"

export CHECKPOINT_FILE
export SWEEP_FILE
export JOB_SCRIPT
export LOG_ROOT

exec "${SCRIPT_DIR}/submit_hope_hierarchical_matrix.sh"
