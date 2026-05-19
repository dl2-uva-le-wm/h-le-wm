#!/bin/bash

# Snellius job: evaluate Hi-LeWM (2-level) on PushT from a local object checkpoint.
#
# Default checkpoint expected:
#   $STABLEWM_HOME/hi_lewm_l2_d25_epoch_1_object.ckpt
#
# Usage:
#   cd jobs/eval/hi/d25
#   sbatch d25_legacy_l2_policy_eval.sh
#
# Optional overrides:
#   sbatch --export=ALL,POLICY=hi_lewm_l2_d25_epoch_1 d25_legacy_l2_policy_eval.sh
#   sbatch --export=ALL,STABLEWM_HOME=/scratch-shared/$USER/stablewm_data d25_legacy_l2_policy_eval.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=hi_eval_d25_legacy_l2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=out/d25_legacy_l2_policy_eval_%j.out
#SBATCH --error=out/d25_legacy_l2_policy_eval_%j.err

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
  echo "Checked PROJECT_ROOT='${PROJECT_ROOT}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', PWD='${PWD}'" >&2
  echo "Submit from repo root or pass PROJECT_ROOT=/path/to/h-le-wm" >&2
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
  echo "Run jobs/setup/setup_env.sh first, or create the environment from environment-gpu.yml" >&2
  exit 2
fi

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
POLICY="${POLICY:-hi_lewm_l2_d25_epoch_1}"
CONFIG_NAME="${CONFIG_NAME:-hi_pusht}"

# Accept both "run_name" and "<run_name>_object.ckpt" for convenience.
if [[ "${POLICY}" == *_object.ckpt ]]; then
  POLICY="${POLICY%_object.ckpt}"
fi

cd "${REPO_ROOT}"

mkdir -p "${STABLEWM_HOME}"

CKPT_OBJECT_PATH="${STABLEWM_HOME}/${POLICY}_object.ckpt"
DATASET_PATH="${STABLEWM_HOME}/pusht_expert_train.h5"

echo "REPO_ROOT=${REPO_ROOT}"
echo "STABLEWM_HOME=${STABLEWM_HOME}"
echo "POLICY=${POLICY}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "Expected checkpoint: ${CKPT_OBJECT_PATH}"
echo "Expected dataset: ${DATASET_PATH}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: missing dataset ${DATASET_PATH}" >&2
  echo "Run setup first, for example:" >&2
  echo "  sbatch --export=ALL,STABLEWM_HOME=${STABLEWM_HOME} jobs/setup/download_pusht.sh" >&2
  exit 3
fi

if [[ ! -f "${CKPT_OBJECT_PATH}" ]]; then
  echo "ERROR: missing object checkpoint ${CKPT_OBJECT_PATH}" >&2
  echo "Available files in ${STABLEWM_HOME}:" >&2
  ls -1 "${STABLEWM_HOME}" >&2 || true
  echo "If needed, pass a different run name, e.g.:" >&2
  echo "  sbatch --export=ALL,POLICY=hi_lewm_l2_d25_epoch_1 d25_legacy_l2_policy_eval.sh" >&2
  exit 4
fi

echo "Starting Hi-LeWM eval..."
python hi_eval.py \
  --config-name="${CONFIG_NAME}" \
  policy="${POLICY}" \
  wm.num_levels=2

echo "Hi-LeWM eval finished."
echo "Result file should be under: ${STABLEWM_HOME}/hi_pusht_results.txt"
