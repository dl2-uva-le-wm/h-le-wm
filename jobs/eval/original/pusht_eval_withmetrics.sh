#!/bin/bash

# Snellius job: evaluate original baseline LeWM code on PushT, plus save a
# per-eval pass/fail manifest for quick failure inspection.
#
# Usage:
#   cd jobs/eval/original
#   sbatch pusht_eval_withmetrics.sh

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=orig_eval_pusht_metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=out/pusht_eval_withmetrics_%j.out
#SBATCH --error=out/pusht_eval_withmetrics_%j.err

set -eo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(cd -- "${SUBMIT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
mkdir -p "${SUBMIT_DIR}/out"

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

cd "${REPO_ROOT}"

if [[ ! -f "original_eval_with_manifest.py" ]]; then
  echo "ERROR: original_eval_with_manifest.py not found at ${REPO_ROOT}" >&2
  echo "Expected script layout: jobs/eval/original/pusht_eval_withmetrics.sh" >&2
  exit 2
fi

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

echo "Starting original eval with per-episode metrics..."
python original_eval_with_manifest.py --config-name="${CONFIG_NAME}" policy="${POLICY}"

echo "Original eval with metrics finished."
echo "Result file should be under: ${STABLEWM_HOME}/$(dirname "${POLICY}")/pusht_results.txt"
echo "Episode manifest should be under: ${STABLEWM_HOME}/$(dirname "${POLICY}")/pusht_results_episodes.tsv"
