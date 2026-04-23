#!/bin/bash

# CPU job: analyze latent action space from a high-level epoch checkpoint.
#
# Usage:
#   cd jobs/analysis/pusht
#   sbatch latent_space_epoch10_cpu.sh
#
# Optional overrides:
#   sbatch --export=ALL,RUN_NAME=hi_lewm_p2_train_hope1_21983875 latent_space_epoch10_cpu.sh
#   sbatch --export=ALL,CHECKPOINT_PATH=/abs/path/model_epoch_10_object.ckpt latent_space_epoch10_cpu.sh
#   sbatch --export=ALL,CHECKPOINT_EPOCH=10,NUM_CHUNKS=20000 latent_space_epoch10_cpu.sh

#SBATCH --partition=rome
#SBATCH --gpus=0
#SBATCH --job-name=latent_pusht_ep10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=latent_space_epoch10_%j.out
#SBATCH --error=latent_space_epoch10_%j.err

set -euo pipefail

resolve_repo_root() {
  local c p
  for c in \
    "${PROJECT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${HOME}/h-lewm" \
    "${HOME}/h-le-wm" \
    "/gpfs/home2/${USER}/h-lewm" \
    "/gpfs/home2/${USER}/h-le-wm"; do
    [[ -z "${c}" ]] && continue
    for p in "${c}" "${c}/.." "${c}/../.." "${c}/../../.."; do
      if p="$(cd "${p}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${p}/hi_train.py" && -f "${p}/tests/analyze_hi_latent_action_space.py" ]]; then
          echo "${p}"
          return 0
        fi
      fi
    done
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root with hi_train.py and tests/analyze_hi_latent_action_space.py" >&2
  exit 2
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

set +u
eval "$(conda shell.bash hook)"
conda activate lewm-gpu
set -u

SCRATCH_STABLEWM_HOME="${SCRATCH_STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
RUNS_ROOT="${RUNS_ROOT:-${SCRATCH_STABLEWM_HOME}/runs}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
RUN_NAME="${RUN_NAME:-}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-10}"
DATASET_NAME="${DATASET_NAME:-pusht_expert_train}"

XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch-shared/${USER}/.cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
mkdir -p "${XDG_CACHE_HOME}" "${XDG_CACHE_HOME}/fontconfig" "${MPLCONFIGDIR}"
export XDG_CACHE_HOME
export MPLCONFIGDIR

NUM_CHUNKS="${NUM_CHUNKS:-12000}"
MIN_CHUNK_LEN="${MIN_CHUNK_LEN:-1}"
MAX_CHUNK_LEN="${MAX_CHUNK_LEN:-15}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_COSINE_PAIRS="${NUM_COSINE_PAIRS:-50000}"
ANALYSIS_SEED="${ANALYSIS_SEED:-3072}"

if [[ -z "${RUN_NAME}" && -z "${CHECKPOINT_PATH}" ]]; then
  RUN_NAME="hi_lewm_p2_train_hope1_21983875"
fi

ANALYSIS_OUT_DIR="${ANALYSIS_OUT_DIR:-${SCRATCH_STABLEWM_HOME}/analysis/latent/${RUN_NAME:-auto}_epoch${CHECKPOINT_EPOCH}_job_${SLURM_JOB_ID:-manual}}"

cd "${REPO_ROOT}"

CMD=(
  python tests/analyze_hi_latent_action_space.py
  --checkpoint-epoch "${CHECKPOINT_EPOCH}"
  --dataset-name "${DATASET_NAME}"
  --cache-dir "${SCRATCH_STABLEWM_HOME}"
  --runs-root "${RUNS_ROOT}"
  --num-chunks "${NUM_CHUNKS}"
  --min-chunk-len "${MIN_CHUNK_LEN}"
  --max-chunk-len "${MAX_CHUNK_LEN}"
  --batch-size "${BATCH_SIZE}"
  --num-cosine-pairs "${NUM_COSINE_PAIRS}"
  --seed "${ANALYSIS_SEED}"
  --output-dir "${ANALYSIS_OUT_DIR}"
)

if [[ -n "${RUN_NAME}" ]]; then
  CMD+=(--run-name "${RUN_NAME}")
fi

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT_PATH}")
fi

echo "Repo root: ${REPO_ROOT}"
echo "Scratch stablewm home: ${SCRATCH_STABLEWM_HOME}"
echo "Runs root: ${RUNS_ROOT}"
echo "Run name: ${RUN_NAME:-<auto latest epoch candidate>}"
echo "Checkpoint path override: ${CHECKPOINT_PATH:-<none>}"
echo "Checkpoint epoch: ${CHECKPOINT_EPOCH}"
echo "Output dir: ${ANALYSIS_OUT_DIR}"
echo "Command:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Latent analysis complete."
echo "Artifacts: ${ANALYSIS_OUT_DIR}"
