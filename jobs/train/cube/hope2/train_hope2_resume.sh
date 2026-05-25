#!/usr/bin/env bash

# Snellius resume job for HOPE2 on OGBench Cube:
# - Resume an existing run in place from its latest weights checkpoint
# - Reuse the same run directory and W&B run id
# - Read dataset + pretrained checkpoint from node-local TMPDIR (scratch-node)
#
# Usage:
#   cd jobs/train/cube
#   sbatch hope2/train_hope2_resume.sh
#
# Defaults:
#   - RESUME_RUN_NAME=hi_lewm_cube_train_hope2_23052898
#   - MAX_EPOCHS=15  (total target epoch count, not additional epochs)
#   - WALLTIME=02:00:00 via the SBATCH header below
#
# Optional overrides:
#   RESUME_RUN_NAME=hi_lewm_cube_train_hope2_23052898 MAX_EPOCHS=15 sbatch hope2/train_hope2_resume.sh
#   WANDB_RUN_ID=run_23052898 sbatch hope2/train_hope2_resume.sh
#   sbatch --time=04:00:00 hope2/train_hope2_resume.sh

#SBATCH --partition=gpu_h100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --job-name=hi_l2_cube_train_hope2_resume
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=output/hope2/train_hope2_resume_%j.out
#SBATCH --error=output/hope2/train_hope2_resume_%j.err

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
        if [[ -f "${p}/h_le_wm/train/hierarchical.py" && -f "${p}/h_le_wm/config/train/hi_lewm.yaml" ]]; then
          echo "${p}"
          return 0
        fi
      fi
    done
  done
  return 1
}

infer_run_suffix() {
  local run_name="$1"
  if [[ "${run_name}" =~ _([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  fi
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root." >&2
  exit 2
fi

BASELINE_ROOT="${REPO_ROOT}/third_party/lewm"
if [[ ! -f "${BASELINE_ROOT}/module.py" || ! -f "${BASELINE_ROOT}/utils.py" ]]; then
  echo "ERROR: Baseline submodule is missing or incomplete at ${BASELINE_ROOT}." >&2
  echo "Expected files: module.py and utils.py" >&2
  echo "Fix on the login node with:" >&2
  echo "  cd ${REPO_ROOT}" >&2
  echo "  git submodule update --init --recursive third_party/lewm" >&2
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
  echo "Set it in ${WANDB_ENV_FILE} or submit with: sbatch --export=ALL,WANDB_API_KEY=<your_key> hope2/train_hope2_resume.sh" >&2
  exit 2
fi
wandb login --relogin "${WANDB_API_KEY}"

WANDB_ENTITY_OVERRIDE="${WANDB_ENTITY:-null}"
WANDB_PROJECT="${WANDB_PROJECT:-hi_lewm}"

######################################## TRAIN SETUP #######################################

SCRATCH_STABLEWM_HOME="${SCRATCH_STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
DATASET_FILE="${DATASET_FILE:-cube_single_expert.h5}"
DATASET_BASENAME="$(basename "${DATASET_FILE}")"
LOCAL_DATASET_REL="${LOCAL_DATASET_REL:-ogbench/${DATASET_BASENAME}}"
CKPT_REL="${CKPT_REL:-cube/lewm_object.ckpt}"
MAX_EPOCHS="${MAX_EPOCHS:-15}"
LATENT_ACTION_DIM="${LATENT_ACTION_DIM:-32}"
RESUME_RUN_NAME="${RESUME_RUN_NAME:-hi_lewm_cube_train_hope2_23052898}"
RESUME_RUN_SUFFIX="$(infer_run_suffix "${RESUME_RUN_NAME}")"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
if [[ -z "${WANDB_RUN_ID}" && -n "${RESUME_RUN_SUFFIX}" ]]; then
  WANDB_RUN_ID="run_${RESUME_RUN_SUFFIX}"
fi
if [[ -z "${WANDB_RUN_ID}" ]]; then
  echo "ERROR: Could not infer WANDB_RUN_ID from RESUME_RUN_NAME='${RESUME_RUN_NAME}'." >&2
  echo "Set it explicitly, e.g. WANDB_RUN_ID=run_23052898 sbatch hope2/train_hope2_resume.sh" >&2
  exit 2
fi

resolve_dataset_path() {
  local candidate
  for candidate in \
    "${SCRATCH_STABLEWM_HOME}/${DATASET_FILE}" \
    "${SCRATCH_STABLEWM_HOME}/ogbench/${DATASET_BASENAME}"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "${SCRATCH_STABLEWM_HOME}/${DATASET_FILE}"
  return 0
}

SRC_DATASET="$(resolve_dataset_path)"
SRC_CKPT="${SCRATCH_STABLEWM_HOME}/${CKPT_REL}"
RUN_DIR="${SCRATCH_STABLEWM_HOME}/runs/${RESUME_RUN_NAME}"
RUN_WEIGHTS_CKPT="${RUN_WEIGHTS_CKPT:-${RUN_DIR}/${RESUME_RUN_NAME}_weights.ckpt}"

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
LOCAL_DATASET="${LOCAL_STABLEWM_HOME}/${LOCAL_DATASET_REL}"
LOCAL_CKPT="${LOCAL_STABLEWM_HOME}/${CKPT_REL}"

# Read data/checkpoint from local scratch for speed.
export STABLEWM_HOME="${LOCAL_STABLEWM_HOME}"

echo "Repo root: ${REPO_ROOT}"
echo "Scratch home: ${SCRATCH_STABLEWM_HOME}"
echo "Local home: ${LOCAL_STABLEWM_HOME}"
echo "STABLEWM_HOME (read path): ${STABLEWM_HOME}"
echo "Resume run name: ${RESUME_RUN_NAME}"
echo "Resume run dir: ${RUN_DIR}"
echo "Resume weights ckpt: ${RUN_WEIGHTS_CKPT}"
echo "TMPDIR: ${TMPDIR}"
echo "Dataset: ${DATASET_FILE}"
echo "Source dataset path: ${SRC_DATASET}"
echo "Local dataset path: ${LOCAL_DATASET}"
echo "Checkpoint: ${CKPT_REL}"
echo "W&B run id: ${WANDB_RUN_ID}"
echo "Max epochs target: ${MAX_EPOCHS}"
echo "Latent action dim: ${LATENT_ACTION_DIM}"
echo "Resume behavior: continue in place from the existing training-state checkpoint."

echo ""
echo "==> Preparing node-local copy in ${LOCAL_STABLEWM_HOME}"
mkdir -p "$(dirname "${LOCAL_DATASET}")" "$(dirname "${LOCAL_CKPT}")"
rsync -ah --info=progress2 "${SRC_DATASET}" "${LOCAL_DATASET}"
rsync -ah --info=progress2 "${SRC_CKPT}" "${LOCAL_CKPT}"

cd "${REPO_ROOT}"

CMD=(
  python -m h_le_wm.train.hierarchical
  data=hi_ogb
  output_model_name="${RESUME_RUN_NAME}"
  subdir="${RUN_DIR}"
  wandb.config.entity="${WANDB_ENTITY_OVERRIDE}"
  wandb.config.project="${WANDB_PROJECT}"
  wandb.config.id="${WANDB_RUN_ID}"
  trainer.max_epochs="${MAX_EPOCHS}"
  wm.high_level.latent_action_dim="${LATENT_ACTION_DIM}"
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
