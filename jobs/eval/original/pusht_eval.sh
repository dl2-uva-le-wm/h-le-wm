#!/bin/bash

# Snellius job: evaluate original baseline LeWM code (submodule) on PushT.
#
# This script explicitly runs:
#   third_party/lewm/eval.py
#
# Usage:
#   sbatch jobs/eval/original/pusht_eval.sh
#
# Optional overrides:
#   sbatch --export=ALL,PROJECT_ROOT=$PWD jobs/eval/original/pusht_eval.sh
#   sbatch --export=ALL,POLICY=pusht/lewm jobs/eval/original/pusht_eval.sh
#   sbatch --export=ALL,STABLEWM_HOME=/scratch-shared/$USER/stablewm_data jobs/eval/original/pusht_eval.sh
#   sbatch --export=ALL,HF_URL=https://huggingface.co/quentinll/lewm-pusht/tree/main jobs/eval/original/pusht_eval.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=orig_eval_pusht
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=pusht_eval_%j.out
#SBATCH --error=pusht_eval_%j.err

set -eo pipefail

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
POLICY="${POLICY:-pusht/lewm}"
CONFIG_NAME="${CONFIG_NAME:-pusht.yaml}"
HF_URL="${HF_URL:-https://huggingface.co/quentinll/lewm-pusht/tree/main}"

resolve_repo_root() {
  local c p
  for c in \
    "${PROJECT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${HOME}/h-lewm" \
    "${HOME}/h-le-wm" \
    "/gpfs/home2/${USER}/h-lewm" \
    "/gpfs/home2/${USER}/h-le-wm" \
    "/gpfs/home3/${USER}/h-lewm" \
    "/gpfs/home3/${USER}/h-le-wm"; do
    [[ -z "${c}" ]] && continue
    for p in "${c}" "${c}/.." "${c}/../.." "${c}/../../.."; do
      if p="$(cd "${p}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${p}/third_party/lewm/eval.py" ]]; then
          echo "${p}"
          return 0
        fi
      fi
    done
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root with third_party/lewm/eval.py" >&2
  echo "Checked: PROJECT_ROOT='${PROJECT_ROOT:-}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', PWD='${PWD:-}', HOME='${HOME:-}'" >&2
  exit 2
fi

cd "${REPO_ROOT}"

mkdir -p jobs/eval/original/out
mkdir -p "${STABLEWM_HOME}"

CKPT_OBJECT_PATH="${STABLEWM_HOME}/${POLICY}_object.ckpt"
DATASET_PATH="${STABLEWM_HOME}/pusht_expert_train.h5"

echo "REPO_ROOT=${REPO_ROOT}"
echo "STABLEWM_HOME=${STABLEWM_HOME}"
echo "POLICY=${POLICY}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "HF_URL=${HF_URL}"
echo "Expected checkpoint: ${CKPT_OBJECT_PATH}"
echo "Expected dataset: ${DATASET_PATH}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: missing dataset ${DATASET_PATH}" >&2
  echo "Run setup first, for example:" >&2
  echo "  sbatch --export=ALL,STABLEWM_HOME=${STABLEWM_HOME} jobs/setup/download_pusht.sh" >&2
  exit 3
fi

if [[ ! -f "${CKPT_OBJECT_PATH}" ]]; then
  echo "Checkpoint object not found. Converting from Hugging Face..."
  python scripts/convert_hf_weights_to_object_ckpt.py \
    --hf-url "${HF_URL}" \
    --run-name "${POLICY}"
fi

echo "Starting ORIGINAL submodule eval..."
cd third_party/lewm
python eval.py --config-name="${CONFIG_NAME}" policy="${POLICY}"

echo "Original eval finished."
echo "Result file should be under: ${STABLEWM_HOME}/$(dirname "${POLICY}")/pusht_results.txt"
