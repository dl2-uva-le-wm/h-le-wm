#!/bin/bash

# Snellius job: Train H-LeWM P2 on PushT with all RC fixes applied.
#
# Key changes vs train.sh:
#   - MAX_EPOCHS=50 (RC-4)
#   - latent_action_dim=32, fixed_stride=1 (RC-2, RC-3)
#   - predictor_high depth=8 / mlp_dim=4096 (RC-5)
#   - preembed_actions=True (RC-6)
#   - lambda_var=0.1 (RC-7)
#
# Usage:
#   cd jobs/2_levels/pusht
#   sbatch train_v2.sh
#
# Overrides:
#   MAX_EPOCHS=100 RUN_NAME=hi_lewm_v2_e100 sbatch train_v2.sh
#   LATENT_DIM=64 sbatch train_v2.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=hi_l2_pusht_v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=06:00:00
#SBATCH --output=train_v2_%j.out
#SBATCH --error=train_v2_%j.err

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
    "/gpfs/home3/${USER}/h-le-wm"; do
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

module purge
module load 2025
module load Anaconda3/2025.06-1

set +u
eval "$(conda shell.bash hook)"
conda activate lewm-gpu
set -u

WANDB_ENV_FILE="${WANDB_ENV_FILE:-$HOME/.config/wandb.env}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  set -a; source "${WANDB_ENV_FILE}"; set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set." >&2
  exit 2
fi
wandb login --relogin "${WANDB_API_KEY}"

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
PRETRAINED_LEWM_CKPT="${PRETRAINED_LEWM_CKPT:-${STABLEWM_HOME}/pusht/lewm_object.ckpt}"

if [[ ! -f "${PRETRAINED_LEWM_CKPT}" ]]; then
  echo "ERROR: pretrained checkpoint not found: ${PRETRAINED_LEWM_CKPT}" >&2
  exit 2
fi

MAX_EPOCHS="${MAX_EPOCHS:-50}"
RUN_NAME="${RUN_NAME:-hi_lewm_v2}"
LATENT_DIM="${LATENT_DIM:-32}"
DEPTH_HIGH="${DEPTH_HIGH:-8}"
MLP_HIGH="${MLP_HIGH:-4096}"
PREEMBED="${PREEMBED:-True}"
LAMBDA_VAR="${LAMBDA_VAR:-0.1}"
STRIDE="${STRIDE:-1}"
NUM_WP="${NUM_WP:-5}"
MAX_SPAN=$(( (NUM_WP - 1) * STRIDE ))

echo "Repo root: ${REPO_ROOT}"
echo "STABLEWM_HOME=${STABLEWM_HOME}"
echo "Run name: ${RUN_NAME}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "latent_action_dim: ${LATENT_DIM}  (was 192)"
echo "predictor_high: depth=${DEPTH_HIGH} mlp_dim=${MLP_HIGH}"
echo "preembed_actions: ${PREEMBED}"
echo "lambda_var: ${LAMBDA_VAR}"
echo "waypoints: num=${NUM_WP} stride=${STRIDE} max_span=${MAX_SPAN}"
echo "Pretrained ckpt: ${PRETRAINED_LEWM_CKPT}"

cd "${REPO_ROOT}"

CMD=(
  python hi_train.py
  data=hi_pusht
  output_model_name="${RUN_NAME}"
  wandb.config.entity="${WANDB_ENTITY:-null}"
  wandb.config.project="${WANDB_PROJECT:-hi_lewm}"
  trainer.max_epochs="${MAX_EPOCHS}"
  training.train_low_level=False
  pretrained_low_level.enabled=True
  pretrained_low_level.checkpoint.selection_mode=explicit_path
  pretrained_low_level.checkpoint.path="${PRETRAINED_LEWM_CKPT}"
  pretrained_low_level.freeze.encoder=True
  pretrained_low_level.freeze.low_level_predictor=True
  pretrained_low_level.freeze.low_level_action_encoder=True
  pretrained_low_level.freeze.projector=True
  pretrained_low_level.freeze.low_pred_proj=True
  pretrained_low_level.freeze.high_pred_proj=False
  wm.high_level.latent_action_dim="${LATENT_DIM}"
  wm.high_level.macro_to_condition_proj=linear
  wm.high_level.preembed_actions="${PREEMBED}"
  wm.high_level.waypoints.strategy=fixed_stride
  wm.high_level.waypoints.num="${NUM_WP}"
  wm.high_level.waypoints.stride="${STRIDE}"
  wm.high_level.waypoints.max_span="${MAX_SPAN}"
  latent_action_encoder.max_seq_len="${STRIDE}"
  predictor_high.depth="${DEPTH_HIGH}"
  predictor_high.mlp_dim="${MLP_HIGH}"
  loss.alpha=0.0
  loss.beta=1.0
  loss.lambda_var="${LAMBDA_VAR}"
)

echo "Launching:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"
