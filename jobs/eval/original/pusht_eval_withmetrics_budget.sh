#!/bin/bash

# Snellius job: evaluate original baseline LeWM code on PushT, plus save a
# per-eval pass/fail manifest for quick failure inspection.
#
# Variant relative to pusht_eval_withmetrics.sh:
# - eval.eval_budget: 50 -> 100
#
# Usage:
#   cd jobs/eval/original
#   sbatch pusht_eval_withmetrics_budget.sh

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1

#SBATCH --job-name=orig_eval_pusht_budget
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=out/pusht_eval_withmetrics_budget_%j.out
#SBATCH --error=out/pusht_eval_withmetrics_budget_%j.err

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
VARIANT_NAME="${VARIANT_NAME:-budget100}"
JOB_TOKEN="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval_original_${VARIANT_NAME}_${JOB_TOKEN}}"
RESULT_FILENAME="${RESULT_FILENAME:-pusht_results_${VARIANT_NAME}.txt}"
EVAL_BUDGET="${EVAL_BUDGET:-100}"
PLAN_HORIZON="${PLAN_HORIZON:-}"

cd "${REPO_ROOT}"

if [[ ! -f "original_eval_with_manifest.py" ]]; then
  echo "ERROR: original_eval_with_manifest.py not found at ${REPO_ROOT}" >&2
  echo "Expected script layout: jobs/eval/original/pusht_eval_withmetrics_budget.sh" >&2
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
echo "VARIANT_NAME=${VARIANT_NAME}"
echo "EVAL_SUBDIR=${EVAL_SUBDIR}"
echo "RESULT_FILENAME=${RESULT_FILENAME}"
if [[ -n "${EVAL_BUDGET}" ]]; then
  echo "Override eval_budget=${EVAL_BUDGET}"
fi
if [[ -n "${PLAN_HORIZON}" ]]; then
  echo "Override plan_config.horizon=${PLAN_HORIZON}"
fi
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
CMD=(
  python original_eval_with_manifest.py
  --config-name="${CONFIG_NAME}"
  "policy=${POLICY}"
  "output.filename=${RESULT_FILENAME}"
  "+output.subdir=${EVAL_SUBDIR}"
)

if [[ -n "${EVAL_BUDGET}" ]]; then
  CMD+=( "eval.eval_budget=${EVAL_BUDGET}" )
fi

if [[ -n "${PLAN_HORIZON}" ]]; then
  CMD+=( "plan_config.horizon=${PLAN_HORIZON}" )
fi

printf '  %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Original eval with metrics finished."
echo "Artifacts should be under: ${STABLEWM_HOME}/$(dirname "${POLICY}")/${EVAL_SUBDIR}"
echo "Result file should be under: ${STABLEWM_HOME}/$(dirname "${POLICY}")/${EVAL_SUBDIR}/${RESULT_FILENAME}"
echo "Episode manifest should be under: ${STABLEWM_HOME}/$(dirname "${POLICY}")/${EVAL_SUBDIR}/${RESULT_FILENAME%.*}_episodes.tsv"
