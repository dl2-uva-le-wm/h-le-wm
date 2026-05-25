#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-${SCRIPT_DIR}/checkpoints_empirical_cem_ablation.txt}"
SWEEP_FILE="${SWEEP_FILE:-${SCRIPT_DIR}/ablation_reducted_matrix_sweep.csv}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval_hope_hierarchical_matrix.sh}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs_empirical_cem_ablation_matrix}"

# Current repo support for the empirical-CEM ablation:
# only the high-level solver is swapped to EmpiricalMacroActionSolver.
# These defaults are taken explicitly from config/eval/hi_pusht.yaml.
export EMPIRICAL_MACRO_ENABLED="${EMPIRICAL_MACRO_ENABLED:-true}"
export EMPIRICAL_MACRO_NUM_SEQUENCES="${EMPIRICAL_MACRO_NUM_SEQUENCES:-65536}"
export EMPIRICAL_MACRO_CHUNK_LEN="${EMPIRICAL_MACRO_CHUNK_LEN:-5}"
export EMPIRICAL_MACRO_RESIDUAL_SCALE="${EMPIRICAL_MACRO_RESIDUAL_SCALE:-0.2}"
export EMPIRICAL_MACRO_MIN_RESIDUAL_STD="${EMPIRICAL_MACRO_MIN_RESIDUAL_STD:-1.0e-3}"
export EMPIRICAL_MACRO_RETURN_TOP_CANDIDATES="${EMPIRICAL_MACRO_RETURN_TOP_CANDIDATES:-8}"
export EMPIRICAL_MACRO_ENCODE_BATCH_SIZE="${EMPIRICAL_MACRO_ENCODE_BATCH_SIZE:-4096}"
export EMPIRICAL_MACRO_STAGE_SAMPLING="${EMPIRICAL_MACRO_STAGE_SAMPLING:-sequence}"

export CHECKPOINT_FILE
export SWEEP_FILE
export JOB_SCRIPT
export LOG_ROOT

exec "${SCRIPT_DIR}/submit_hope_hierarchical_matrix.sh"
