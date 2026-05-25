#!/usr/bin/env bash

# Snellius training job for the frozen HOPE2 decoder probe:
# - Read dataset + frozen HOPE2 object checkpoint from node-local TMPDIR
# - Train only the decoder probe with W&B logging
# - Save probe artifacts directly to shared scratch
#
# Usage:
#   cd jobs/train/pusht
#   sbatch hope2/train_decoder_probe.sh
#
# Optional overrides:
#   MODE=true_only sbatch hope2/train_decoder_probe.sh
#   MODE=pred_exposed INIT_DECODER_CKPT=/scratch-shared/$USER/stablewm_data/runs/.../hi_decoder_probe_true_probe.pt sbatch hope2/train_decoder_probe.sh
#   TRAIN_RUN_NAME=hi_decoder_probe_true_custom sbatch hope2/train_decoder_probe.sh

#SBATCH --partition=gpu_h100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --job-name=hi_decoder_probe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --output=output/hope2/train_decoder_probe_%j.out
#SBATCH --error=output/hope2/train_decoder_probe_%j.err

set -euo pipefail

resolve_repo_root() {
  local c p
  for c in \
    "${PROJECT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${HOME}/h-le-wm" \
    "${HOME}/h-lewm" \
    "/gpfs/home2/${USER}/h-le-wm" \
    "/gpfs/home2/${USER}/h-lewm"; do
    [[ -z "${c}" ]] && continue
    for p in "${c}" "${c}/.." "${c}/../.." "${c}/../../.."; do
      if p="$(cd "${p}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${p}/h_le_wm/probe/train.py" && -f "${p}/h_le_wm/config/train/hi_decoder_probe.yaml" ]]; then
          echo "${p}"
          return 0
        fi
      fi
    done
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root." >&2
  exit 2
fi

BASELINE_ROOT="${REPO_ROOT}/third_party/lewm"
if [[ ! -f "${BASELINE_ROOT}/module.py" || ! -f "${BASELINE_ROOT}/utils.py" ]]; then
  echo "ERROR: Baseline submodule is missing or incomplete at ${BASELINE_ROOT}." >&2
  echo "Fix on the login node with:" >&2
  echo "  cd ${REPO_ROOT}" >&2
  echo "  git submodule update --init --recursive third_party/lewm" >&2
  exit 2
fi

if [[ -z "${TMPDIR:-}" ]]; then
  echo "ERROR: TMPDIR is not set." >&2
  exit 2
fi
if [[ "${TMPDIR}" != /scratch-node/* ]]; then
  echo "ERROR: TMPDIR is '${TMPDIR}', expected /scratch-node/... for node-local training." >&2
  exit 2
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

set +u
eval "$(conda shell.bash hook)"
conda activate lewm-gpu
set -u

WANDB_ENV_FILE="${WANDB_ENV_FILE:-$HOME/.config/wandb.env}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  set -a
  source "${WANDB_ENV_FILE}"
  set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set." >&2
  exit 2
fi
wandb login --relogin "${WANDB_API_KEY}"

WANDB_ENTITY_OVERRIDE="${WANDB_ENTITY:-null}"
WANDB_PROJECT="${WANDB_PROJECT:-hi_decoder_probe}"

SCRATCH_STABLEWM_HOME="${SCRATCH_STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
DATASET_FILE="${DATASET_FILE:-pusht_expert_train.h5}"
HOPE2_CKPT_REL="${HOPE2_CKPT_REL:-runs/hi_lewm_p2_train_hope2_22253175/hi_lewm_p2_train_hope2_22253175_epoch_15_object.ckpt}"
MODE="${MODE:-true_only}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-hi_decoder_probe_${MODE}_${SLURM_JOB_ID:-manual}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-run_${SLURM_JOB_ID:-manual}}"
IMAGE_LOG_INTERVAL="${IMAGE_LOG_INTERVAL:-1}"
INIT_DECODER_CKPT="${INIT_DECODER_CKPT:-}"
RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"

if [[ "${MODE}" != "true_only" && "${MODE}" != "pred_exposed" ]]; then
  echo "ERROR: MODE must be one of: true_only, pred_exposed" >&2
  exit 2
fi

SRC_DATASET="${SCRATCH_STABLEWM_HOME}/${DATASET_FILE}"
SRC_HOPE2_CKPT="${SCRATCH_STABLEWM_HOME}/${HOPE2_CKPT_REL}"
if [[ ! -f "${SRC_DATASET}" ]]; then
  echo "ERROR: dataset file not found: ${SRC_DATASET}" >&2
  exit 2
fi
if [[ ! -f "${SRC_HOPE2_CKPT}" ]]; then
  echo "ERROR: HOPE2 checkpoint not found: ${SRC_HOPE2_CKPT}" >&2
  exit 2
fi

LOCAL_STABLEWM_HOME="${LOCAL_STABLEWM_HOME:-${TMPDIR}/${USER}_stablewm_data_${SLURM_JOB_ID:-manual}}"
LOCAL_DATASET="${LOCAL_STABLEWM_HOME}/${DATASET_FILE}"
LOCAL_HOPE2_CKPT="${LOCAL_STABLEWM_HOME}/${HOPE2_CKPT_REL}"
PERSIST_RUN_DIR="${PERSIST_RUN_DIR:-${SCRATCH_STABLEWM_HOME}/runs/${TRAIN_RUN_NAME}}"

export STABLEWM_HOME="${LOCAL_STABLEWM_HOME}"

echo "Repo root: ${REPO_ROOT}"
echo "Scratch home: ${SCRATCH_STABLEWM_HOME}"
echo "Local home: ${LOCAL_STABLEWM_HOME}"
echo "Output run dir (shared): ${PERSIST_RUN_DIR}"
echo "TMPDIR: ${TMPDIR}"
echo "Dataset: ${DATASET_FILE}"
echo "Frozen HOPE2 checkpoint: ${HOPE2_CKPT_REL}"
echo "Mode: ${MODE}"
echo "Run name: ${TRAIN_RUN_NAME}"
echo "W&B run id: ${WANDB_RUN_ID}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Image log interval: ${IMAGE_LOG_INTERVAL}"
if [[ -n "${INIT_DECODER_CKPT}" ]]; then
  echo "Init decoder checkpoint: ${INIT_DECODER_CKPT}"
fi
if [[ -n "${RESUME_FROM_CKPT}" ]]; then
  echo "Resume checkpoint: ${RESUME_FROM_CKPT}"
fi

mkdir -p "$(dirname "${LOCAL_DATASET}")" "$(dirname "${LOCAL_HOPE2_CKPT}")" "${PERSIST_RUN_DIR}"
rsync -ah --info=progress2 "${SRC_DATASET}" "${LOCAL_DATASET}"
rsync -ah --info=progress2 "${SRC_HOPE2_CKPT}" "${LOCAL_HOPE2_CKPT}"

cd "${REPO_ROOT}"

CMD=(
  python -m h_le_wm.probe.train
  data=hi_pusht
  output_model_name="${TRAIN_RUN_NAME}"
  subdir="${PERSIST_RUN_DIR}"
  trainer.max_epochs="${MAX_EPOCHS}"
  probe.mode="${MODE}"
  probe.checkpoint.path="${LOCAL_HOPE2_CKPT}"
  wandb.config.entity="${WANDB_ENTITY_OVERRIDE}"
  wandb.config.project="${WANDB_PROJECT}"
  wandb.config.id="${WANDB_RUN_ID}"
  wandb.image_log_interval="${IMAGE_LOG_INTERVAL}"
)

if [[ -n "${INIT_DECODER_CKPT}" ]]; then
  CMD+=("probe.init_decoder_checkpoint=${INIT_DECODER_CKPT}")
fi
if [[ -n "${RESUME_FROM_CKPT}" ]]; then
  CMD+=("checkpointing.resume_from=${RESUME_FROM_CKPT}")
fi
if [[ -n "${LIMIT_TRAIN_BATCHES}" ]]; then
  CMD+=("trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES}")
fi
if [[ -n "${LIMIT_VAL_BATCHES}" ]]; then
  CMD+=("trainer.limit_val_batches=${LIMIT_VAL_BATCHES}")
fi

echo ""
echo "==> Launching training command:"
printf '  %q' "${CMD[@]}"
echo

SECONDS=0
"${CMD[@]}"
elapsed="${SECONDS}"

echo ""
echo "Decoder probe training finished in ${elapsed}s."
echo "Artifacts are stored in: ${PERSIST_RUN_DIR}"
