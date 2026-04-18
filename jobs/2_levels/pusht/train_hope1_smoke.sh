#!/bin/bash

# Snellius training job (simple):
# - Read dataset + pretrained checkpoint from node-local TMPDIR (scratch-node)
# - Save training artifacts directly to shared scratch
#
# Usage:
#   cd jobs/2_levels/pusht
#   sbatch train_hope1_smoke.sh
#
# Optional overrides:
#   MAX_EPOCHS=1 sbatch train_hope1_smoke.sh
#   TRAIN_RUN_NAME=hi_lewm_p2_train_hope1_smoke_custom sbatch train_hope1_smoke.sh
#   SCRATCH_STABLEWM_HOME=/scratch-shared/$USER/stablewm_data sbatch train_hope1_smoke.sh

#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --job-name=hi_l2_pusht_train_hope1_smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=train_hope1_smoke_%j.out
#SBATCH --error=train_hope1_smoke_%j.err

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
        if [[ -f "${p}/hi_train.py" && -f "${p}/config/train/hi_lewm.yaml" ]]; then
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

if [[ -z "${TMPDIR:-}" ]]; then
  echo "ERROR: TMPDIR is not set." >&2
  echo "Expected a scratch-node allocation where TMPDIR points under /scratch-node." >&2
  exit 2
fi
if [[ "${TMPDIR}" != /scratch-node/* ]]; then
  echo "ERROR: TMPDIR is '${TMPDIR}', expected /scratch-node/... for node-local training." >&2
  echo "Make sure this job is submitted with '#SBATCH --constraint=scratch-node'." >&2
  exit 2
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

# Some cluster conda activation scripts reference unset vars; keep strict mode elsewhere.
set +u
eval "$(conda shell.bash hook)"
conda activate lewm-gpu
set -u

####################################### WANDB SETUP #######################################
WANDB_ENV_FILE="${WANDB_ENV_FILE:-$HOME/.config/wandb.env}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  set -a
  source "${WANDB_ENV_FILE}"
  set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set." >&2
  echo "Set it in ${WANDB_ENV_FILE} or submit with: sbatch --export=ALL,WANDB_API_KEY=<your_key> train_hope1_smoke.sh" >&2
  exit 2
fi
wandb login --relogin "${WANDB_API_KEY}"

WANDB_ENTITY_OVERRIDE="${WANDB_ENTITY:-null}"
WANDB_PROJECT="${WANDB_PROJECT:-hi_lewm}"

######################################## TRAIN SETUP #######################################

SCRATCH_STABLEWM_HOME="${SCRATCH_STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
DATASET_FILE="${DATASET_FILE:-pusht_expert_train.h5}"
CKPT_REL="${CKPT_REL:-pusht/lewm_object.ckpt}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-hi_lewm_p2_train_hope1_smoke_${SLURM_JOB_ID:-manual}}"

SRC_DATASET="${SCRATCH_STABLEWM_HOME}/${DATASET_FILE}"
SRC_CKPT="${SCRATCH_STABLEWM_HOME}/${CKPT_REL}"
if [[ ! -f "${SRC_DATASET}" ]]; then
  echo "ERROR: dataset file not found: ${SRC_DATASET}" >&2
  exit 2
fi
if [[ ! -f "${SRC_CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${SRC_CKPT}" >&2
  exit 2
fi

LOCAL_STABLEWM_HOME="${LOCAL_STABLEWM_HOME:-${TMPDIR}/${USER}_stablewm_data_${SLURM_JOB_ID:-manual}}"
LOCAL_DATASET="${LOCAL_STABLEWM_HOME}/${DATASET_FILE}"
LOCAL_CKPT="${LOCAL_STABLEWM_HOME}/${CKPT_REL}"
PERSIST_RUN_DIR="${PERSIST_RUN_DIR:-${SCRATCH_STABLEWM_HOME}/runs/${TRAIN_RUN_NAME}}"

# Read data/checkpoint from local scratch for speed.
export STABLEWM_HOME="${LOCAL_STABLEWM_HOME}"

echo "Repo root: ${REPO_ROOT}"
echo "Scratch home: ${SCRATCH_STABLEWM_HOME}"
echo "Local home: ${LOCAL_STABLEWM_HOME}"
echo "STABLEWM_HOME (read path): ${STABLEWM_HOME}"
echo "Output run dir (shared): ${PERSIST_RUN_DIR}"
echo "TMPDIR: ${TMPDIR}"
echo "Dataset: ${DATASET_FILE}"
echo "Checkpoint: ${CKPT_REL}"
echo "Run name: ${TRAIN_RUN_NAME}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Early stopping: disabled (no early stopping callback configured)."

echo ""
echo "==> Preparing node-local copy in ${LOCAL_STABLEWM_HOME}"
mkdir -p "$(dirname "${LOCAL_DATASET}")" "$(dirname "${LOCAL_CKPT}")" "${PERSIST_RUN_DIR}"
rsync -ah --info=progress2 "${SRC_DATASET}" "${LOCAL_DATASET}"
rsync -ah --info=progress2 "${SRC_CKPT}" "${LOCAL_CKPT}"

cd "${REPO_ROOT}"

CMD=(
  python hi_train.py
  data=hi_pusht
  output_model_name="${TRAIN_RUN_NAME}"
  subdir="${PERSIST_RUN_DIR}"
  wandb.config.entity="${WANDB_ENTITY_OVERRIDE}"
  wandb.config.project="${WANDB_PROJECT}"
  trainer.max_epochs="${MAX_EPOCHS}"
  training.train_low_level=False
  pretrained_low_level.enabled=True
  pretrained_low_level.checkpoint.selection_mode=explicit_path
  pretrained_low_level.checkpoint.path="${LOCAL_CKPT}"
  pretrained_low_level.freeze.encoder=True
  pretrained_low_level.freeze.low_level_predictor=True
  pretrained_low_level.freeze.low_level_action_encoder=True
  pretrained_low_level.freeze.projector=True
  pretrained_low_level.freeze.low_pred_proj=True
  pretrained_low_level.freeze.high_pred_proj=False
  loss.alpha=0.0
  loss.beta=1.0
)

echo ""
echo "==> Launching training command:"
printf '  %q' "${CMD[@]}"
echo

SECONDS=0
"${CMD[@]}"
elapsed="${SECONDS}"

echo ""
echo "Training finished in ${elapsed}s."
echo "Artifacts are stored in: ${PERSIST_RUN_DIR}"
