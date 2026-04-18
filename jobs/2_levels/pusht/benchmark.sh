#!/bin/bash

# Snellius short benchmark: hierarchical PushT P2 only.
# Usage (from this folder):
#   cd jobs/2_levels/pusht
#   sbatch benchmark.sh
# Optional env overrides:
#   BENCH_STEPS=500 BENCH_LIMIT_VAL_BATCHES=0 BENCH_RUN_NAME=hi_lewm_p2_bench_test sbatch benchmark.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=hi_l2_pusht_bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=bench_p2_%j.out
#SBATCH --error=bench_p2_%j.err

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
  echo "Checked PROJECT_ROOT='${PROJECT_ROOT:-}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', PWD='${PWD:-}'" >&2
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
  echo "Set it in ${WANDB_ENV_FILE} or submit with: sbatch --export=ALL,WANDB_API_KEY=<your_key> benchmark.sh" >&2
  exit 2
fi
wandb login --relogin "${WANDB_API_KEY}"

WANDB_ENTITY_OVERRIDE="${WANDB_ENTITY:-null}"
WANDB_PROJECT="${WANDB_PROJECT:-hi_lewm}"

######################################## BENCH SETUP #######################################

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
PRETRAINED_LEWM_CKPT="${PRETRAINED_LEWM_CKPT:-${STABLEWM_HOME}/pusht/lewm_object.ckpt}"
BENCH_STEPS="${BENCH_STEPS:-500}"
BENCH_LIMIT_VAL_BATCHES="${BENCH_LIMIT_VAL_BATCHES:-0}"
BENCH_MAX_EPOCHS="${BENCH_MAX_EPOCHS:-9999}"
BENCH_RUN_NAME="${BENCH_RUN_NAME:-hi_lewm_p2_bench}"

if [[ ! -f "${PRETRAINED_LEWM_CKPT}" ]]; then
  echo "ERROR: pretrained checkpoint not found: ${PRETRAINED_LEWM_CKPT}" >&2
  exit 2
fi

if ! [[ "${BENCH_STEPS}" =~ ^[0-9]+$ ]] || [[ "${BENCH_STEPS}" -le 0 ]]; then
  echo "ERROR: BENCH_STEPS must be a positive integer (got '${BENCH_STEPS}')" >&2
  exit 2
fi

echo "Repo root: ${REPO_ROOT}"
echo "STABLEWM_HOME=${STABLEWM_HOME}"
echo "W&B entity: ${WANDB_ENTITY:-<default from login>}"
echo "W&B project: ${WANDB_PROJECT}"
echo "Benchmark run name: ${BENCH_RUN_NAME}"
echo "Benchmark max_steps: ${BENCH_STEPS}"
echo "Benchmark limit_val_batches: ${BENCH_LIMIT_VAL_BATCHES}"
echo "Pretrained ckpt: ${PRETRAINED_LEWM_CKPT}"

cd "${REPO_ROOT}"

CMD=(
  python hi_train.py
  data=hi_pusht
  output_model_name="${BENCH_RUN_NAME}"
  wandb.config.entity="${WANDB_ENTITY_OVERRIDE}"
  wandb.config.project="${WANDB_PROJECT}"
  trainer.max_epochs="${BENCH_MAX_EPOCHS}"
  +trainer.max_steps="${BENCH_STEPS}"
  +trainer.limit_val_batches="${BENCH_LIMIT_VAL_BATCHES}"
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
  loss.alpha=0.0
  loss.beta=1.0
)

echo "Launching benchmark command:"
printf '  %q' "${CMD[@]}"
echo

SECONDS=0
"${CMD[@]}"
elapsed="${SECONDS}"
echo "Benchmark finished in ${elapsed}s for ${BENCH_STEPS} train steps."
if [[ "${elapsed}" -gt 0 ]]; then
  python - <<PY
steps = int(${BENCH_STEPS})
sec = int(${elapsed})
print(f"Approx throughput: {steps/sec:.2f} steps/s ({sec/steps:.2f} s/step)")
PY
fi
