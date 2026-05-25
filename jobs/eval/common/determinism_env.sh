#!/usr/bin/env bash

set -euo pipefail

setup_eval_determinism_env() {
  local repo_root="$1"
  local eval_seed="${2:-42}"
  local eval_determinism="${3:-strict}"
  local bootstrap_dir="${repo_root}/jobs/eval/common/python_bootstrap"

  export EVAL_SEED="${eval_seed}"
  export EVAL_DETERMINISM="${eval_determinism}"
  export PYTHONHASHSEED="${PYTHONHASHSEED:-${EVAL_SEED}}"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${bootstrap_dir}:${repo_root}:${PYTHONPATH}"
  else
    export PYTHONPATH="${bootstrap_dir}:${repo_root}"
  fi
}

print_eval_determinism_env() {
  echo "Eval seed: ${EVAL_SEED:-<unset>}"
  echo "Determinism mode: ${EVAL_DETERMINISM:-<unset>}"
  echo "PYTHONHASHSEED: ${PYTHONHASHSEED:-<unset>}"
  echo "CUBLAS_WORKSPACE_CONFIG: ${CUBLAS_WORKSPACE_CONFIG:-<unset>}"
}
