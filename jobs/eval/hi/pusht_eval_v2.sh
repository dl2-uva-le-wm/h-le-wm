#!/bin/bash

# Snellius job: evaluate Hi-LeWM on PushT with RC-1/RC-2 planning fixes.
#
# Key changes vs pusht_eval_l2_d25.sh:
#   - LOW_HORIZON=2 (RC-1: was 5; matches subgoal timing)
#   - HIGH_SAMPLES=300, HIGH_STEPS=15 (RC-2: 32-D needs fewer samples)
#
# Usage:
#   cd jobs/eval/hi
#   sbatch --export=ALL,POLICY=hi_lewm_v2_epoch_50 pusht_eval_v2.sh
#
# Horizon sweep (no retraining needed):
#   LOW_HORIZON=1 sbatch --export=ALL,POLICY=<ckpt> pusht_eval_v2.sh
#   LOW_HORIZON=2 sbatch --export=ALL,POLICY=<ckpt> pusht_eval_v2.sh
#   LOW_HORIZON=5 sbatch --export=ALL,POLICY=<ckpt> pusht_eval_v2.sh  # baseline comparison

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=hi_eval_v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=out/hi_eval_v2_%j.out
#SBATCH --error=out/hi_eval_v2_%j.err

set -eo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${SUBMIT_DIR}/out"

PROJECT_ROOT="${PROJECT_ROOT:-}"

resolve_repo_root() {
  local candidate
  for candidate in "${PROJECT_ROOT}" "${SUBMIT_DIR}" "${SUBMIT_DIR}/../../.."; do
    [[ -n "${candidate}" ]] || continue
    candidate="$(cd -- "${candidate}" >/dev/null 2>&1 && pwd || true)"
    if [[ -n "${candidate}" && -f "${candidate}/hi_eval.py" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root with hi_eval.py" >&2
  exit 2
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
if conda env list | grep -E '(^|[[:space:]])lewm-gpu([[:space:]]|$)' >/dev/null 2>&1; then
  conda activate lewm-gpu
elif conda env list | grep -E '(^|[[:space:]])lewm([[:space:]]|$)' >/dev/null 2>&1; then
  conda activate lewm
else
  echo "ERROR: Could not find conda environment 'lewm-gpu' or 'lewm'" >&2
  exit 2
fi

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
POLICY="${POLICY:-hi_lewm_v2_epoch_50}"
CONFIG_NAME="${CONFIG_NAME:-hi_pusht}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"

# RC-1: low horizon fix; RC-2: reduced CEM budget for 32-D space
LOW_HORIZON="${LOW_HORIZON:-2}"
HIGH_SAMPLES="${HIGH_SAMPLES:-300}"
HIGH_STEPS="${HIGH_STEPS:-15}"
LOW_SAMPLES="${LOW_SAMPLES:-300}"
LOW_STEPS="${LOW_STEPS:-30}"

if [[ "${POLICY}" == *_object.ckpt ]]; then
  POLICY="${POLICY%_object.ckpt}"
fi

cd "${REPO_ROOT}"

CKPT_OBJECT_PATH="${STABLEWM_HOME}/${POLICY}_object.ckpt"
DATASET_PATH="${STABLEWM_HOME}/pusht_expert_train.h5"

echo "REPO_ROOT=${REPO_ROOT}"
echo "STABLEWM_HOME=${STABLEWM_HOME}"
echo "POLICY=${POLICY}"
echo "eval_budget=${EVAL_BUDGET}"
echo "low_horizon=${LOW_HORIZON}  (RC-1 fix: was 5)"
echo "high_solver: num_samples=${HIGH_SAMPLES} n_steps=${HIGH_STEPS}  (RC-2 fix)"
echo "Expected checkpoint: ${CKPT_OBJECT_PATH}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: missing dataset ${DATASET_PATH}" >&2
  exit 3
fi

if [[ ! -f "${CKPT_OBJECT_PATH}" ]]; then
  echo "ERROR: missing object checkpoint ${CKPT_OBJECT_PATH}" >&2
  ls -1 "${STABLEWM_HOME}" >&2 || true
  exit 4
fi

python hi_eval.py \
  --config-name="${CONFIG_NAME}" \
  policy="${POLICY}" \
  eval.eval_budget="${EVAL_BUDGET}" \
  "planning.low.plan_config.horizon=${LOW_HORIZON}" \
  "planning.high.solver.num_samples=${HIGH_SAMPLES}" \
  "planning.high.solver.n_steps=${HIGH_STEPS}" \
  "planning.low.solver.num_samples=${LOW_SAMPLES}" \
  "planning.low.solver.n_steps=${LOW_STEPS}"

echo "Eval done. Results: ${STABLEWM_HOME}/hi_pusht_results.txt"
