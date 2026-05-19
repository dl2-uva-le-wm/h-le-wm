#!/bin/bash

# Snellius training resume job:
# - Resume an existing joint P1+P2 run in place from its latest weights ckpt
# - Reuse the same run directory and W&B run id
# - Keep per-epoch object and training-state checkpoints
#
# Usage:
#   cd jobs/train/pusht
#   sbatch train_joint_levels_resume.sh
#
# Defaults:
#   - RESUME_RUN_NAME=hi_lewm_joint_22368719
#   - MAX_EPOCHS=25  (total target epoch count, not additional epochs)
#   - Keeps latent_action_dim=32, batch=16, grad_acc=8, sigreg=0.2
#
# Optional overrides:
#   RESUME_RUN_NAME=hi_lewm_joint_22368719 MAX_EPOCHS=30 sbatch train_joint_levels_resume.sh
#   BATCH_SIZE=8 ACCUMULATE_GRAD_BATCHES=16 sbatch train_joint_levels_resume.sh

#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --job-name=hi_joint_pusht_resume
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=24:00:00
#SBATCH --output=train_joint_levels_resume_%j.out
#SBATCH --error=train_joint_levels_resume_%j.err

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

BASELINE_ROOT="${REPO_ROOT}/third_party/lewm"
if [[ ! -f "${BASELINE_ROOT}/module.py" || ! -f "${BASELINE_ROOT}/utils.py" ]]; then
  echo "ERROR: Baseline submodule is missing or incomplete at ${BASELINE_ROOT}." >&2
  echo "Run: git submodule update --init --recursive third_party/lewm" >&2
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

####################################### WANDB SETUP #######################################
WANDB_ENV_FILE="${WANDB_ENV_FILE:-$HOME/.config/wandb.env}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  set -a
  source "${WANDB_ENV_FILE}"
  set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set." >&2
  echo "Set it in ${WANDB_ENV_FILE} or submit with: sbatch --export=ALL,WANDB_API_KEY=<your_key> train_joint_levels_resume.sh" >&2
  exit 2
fi
wandb login --relogin "${WANDB_API_KEY}"

WANDB_ENTITY_OVERRIDE="${WANDB_ENTITY:-null}"
WANDB_PROJECT="${WANDB_PROJECT:-hi_lewm}"

######################################## TRAIN SETUP #######################################

SCRATCH_STABLEWM_HOME="${SCRATCH_STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
DATASET_FILE="${DATASET_FILE:-pusht_expert_train.h5}"
CKPT_REL="${CKPT_REL:-pusht/lewm_object.ckpt}"
RESUME_RUN_NAME="${RESUME_RUN_NAME:-hi_lewm_joint_22368719}"
MAX_EPOCHS="${MAX_EPOCHS:-25}"
LATENT_ACTION_DIM="${LATENT_ACTION_DIM:-32}"
OPTIMIZER_LR="${OPTIMIZER_LR:-5e-5}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
TRAINER_PRECISION="${TRAINER_PRECISION:-bf16-mixed}"

SRC_DATASET="${SCRATCH_STABLEWM_HOME}/${DATASET_FILE}"
SRC_CKPT="${SCRATCH_STABLEWM_HOME}/${CKPT_REL}"
RUN_DIR="${SCRATCH_STABLEWM_HOME}/runs/${RESUME_RUN_NAME}"
RUN_WEIGHTS_CKPT="${RUN_DIR}/${RESUME_RUN_NAME}_weights.ckpt"

if [[ ! -f "${SRC_DATASET}" ]]; then
  echo "ERROR: dataset file not found: ${SRC_DATASET}" >&2
  exit 2
fi
if [[ ! -f "${SRC_CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${SRC_CKPT}" >&2
  exit 2
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run directory not found: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${RUN_WEIGHTS_CKPT}" ]]; then
  echo "ERROR: resume weights checkpoint not found: ${RUN_WEIGHTS_CKPT}" >&2
  exit 2
fi

LOCAL_STABLEWM_HOME="${LOCAL_STABLEWM_HOME:-${TMPDIR}/${USER}_stablewm_data_${SLURM_JOB_ID:-manual}}"
LOCAL_DATASET="${LOCAL_STABLEWM_HOME}/${DATASET_FILE}"
LOCAL_CKPT="${LOCAL_STABLEWM_HOME}/${CKPT_REL}"

export STABLEWM_HOME="${LOCAL_STABLEWM_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Repo root: ${REPO_ROOT}"
echo "Scratch home: ${SCRATCH_STABLEWM_HOME}"
echo "Local home: ${LOCAL_STABLEWM_HOME}"
echo "STABLEWM_HOME (read path): ${STABLEWM_HOME}"
echo "Resume run name: ${RESUME_RUN_NAME}"
echo "Resume run dir: ${RUN_DIR}"
echo "Resume weights ckpt: ${RUN_WEIGHTS_CKPT}"
echo "Max epochs target: ${MAX_EPOCHS}"
echo "Latent action dim: ${LATENT_ACTION_DIM}"
echo "Optimizer lr: ${OPTIMIZER_LR}"
echo "SIGReg weight: ${SIGREG_WEIGHT}"
echo "Micro-batch size: ${BATCH_SIZE}"
echo "Gradient accumulation: ${ACCUMULATE_GRAD_BATCHES}"
echo "Trainer precision: ${TRAINER_PRECISION}"
echo "W&B run id: ${RESUME_RUN_NAME}"

echo ""
echo "==> Preparing node-local copy in ${LOCAL_STABLEWM_HOME}"
mkdir -p "$(dirname "${LOCAL_DATASET}")" "$(dirname "${LOCAL_CKPT}")"
rsync -ah --info=progress2 "${SRC_DATASET}" "${LOCAL_DATASET}"
rsync -ah --info=progress2 "${SRC_CKPT}" "${LOCAL_CKPT}"

cd "${REPO_ROOT}"

CMD=(
  python hi_train.py
  data=hi_pusht
  output_model_name="${RESUME_RUN_NAME}"
  subdir="${RUN_DIR}"
  wandb.config.entity="${WANDB_ENTITY_OVERRIDE}"
  wandb.config.project="${WANDB_PROJECT}"
  wandb.config.id="${RESUME_RUN_NAME}"
  trainer.max_epochs="${MAX_EPOCHS}"
  trainer.precision="${TRAINER_PRECISION}"
  +trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}"
  loader.batch_size="${BATCH_SIZE}"
  optimizer.lr="${OPTIMIZER_LR}"
  wm.high_level.latent_action_dim="${LATENT_ACTION_DIM}"
  training.train_low_level=True
  pretrained_low_level.enabled=True
  pretrained_low_level.checkpoint.selection_mode=explicit_path
  pretrained_low_level.checkpoint.path="${LOCAL_CKPT}"
  pretrained_low_level.freeze.encoder=False
  pretrained_low_level.freeze.low_level_predictor=False
  pretrained_low_level.freeze.low_level_action_encoder=False
  pretrained_low_level.freeze.projector=False
  pretrained_low_level.freeze.low_pred_proj=False
  pretrained_low_level.freeze.high_pred_proj=False
  loss.alpha=1.0
  loss.beta=1.0
  loss.sigreg.weight="${SIGREG_WEIGHT}"
  checkpointing.object_dump.epoch_interval=1
  checkpointing.weights_dump.enabled=True
  checkpointing.weights_dump.epoch_interval=1
)

echo ""
echo "==> Launching resume command:"
printf '  %q' "${CMD[@]}"
echo

SECONDS=0
"${CMD[@]}"
elapsed="${SECONDS}"

echo ""
echo "Resume training finished in ${elapsed}s."
echo "Artifacts are stored in: ${RUN_DIR}"
