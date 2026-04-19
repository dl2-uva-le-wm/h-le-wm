#!/bin/bash

# Snellius eval job for Hi-LeWM (2-level) on PushT (short, d=25).
#
# Default behavior:
# - Uses your recent run: hi_lewm_p2_train_hope1_21983875
# - Auto-selects latest object checkpoint in that run directory
# - Evaluates with hi_eval.py --config-name=hi_pusht
# - Sets eval.goal_offset_steps=25 (short setting)
#
# Usage:
#   cd jobs/eval/hi
#   sbatch eval_hope1_short.sh
#
# Common overrides:
#   sbatch --export=ALL,CHECKPOINT_EPOCH=8 eval_hope1_short.sh
#   sbatch --export=ALL,RUN_NAME=hi_lewm_p2_train_hope1_21983875,CHECKPOINT_EPOCH=latest eval_hope1_short.sh
#   sbatch --export=ALL,RESULT_FILENAME=my_eval_short_epoch8.txt eval_hope1_short.sh
#   sbatch --export=ALL,GOAL_OFFSET_STEPS=25 eval_hope1_short.sh
#   sbatch --export=ALL,STABLEWM_HOME=/scratch-shared/$USER/stablewm_data eval_hope1_short.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=hi_eval_hope1_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=eval_hope1_short_%j.out
#SBATCH --error=eval_hope1_short_%j.err

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
        if [[ -f "${p}/hi_eval.py" && -f "${p}/config/eval/hi_pusht.yaml" ]]; then
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
  echo "Submit from repo root or pass PROJECT_ROOT=/path/to/h-le-wm" >&2
  exit 2
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

# Some cluster conda activation scripts reference unset vars; keep strict mode elsewhere.
set +u
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
set -u

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
RUN_NAME="${RUN_NAME:-hi_lewm_p2_train_hope1_21983875}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-latest}"  # "latest" or integer >= 1
CONFIG_NAME="${CONFIG_NAME:-hi_pusht}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"

RUN_DIR="${STABLEWM_HOME}/runs/${RUN_NAME}"
DATASET_PATH="${STABLEWM_HOME}/pusht_expert_train.h5"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: dataset file not found: ${DATASET_PATH}" >&2
  echo "Run setup first, for example:" >&2
  echo "  sbatch --export=ALL,STABLEWM_HOME=${STABLEWM_HOME} jobs/setup/download_pusht.sh" >&2
  exit 3
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run directory not found: ${RUN_DIR}" >&2
  exit 4
fi

if [[ "${CHECKPOINT_EPOCH}" == "latest" ]]; then
  shopt -s nullglob
  candidates=( "${RUN_DIR}/${RUN_NAME}"_epoch_*_object.ckpt )
  shopt -u nullglob
  if (( ${#candidates[@]} == 0 )); then
    echo "ERROR: no object checkpoints found in ${RUN_DIR}" >&2
    echo "Expected pattern: ${RUN_NAME}_epoch_*_object.ckpt" >&2
    exit 5
  fi
  mapfile -t sorted_candidates < <(printf '%s\n' "${candidates[@]}" | sort -V)
  CKPT_OBJECT_PATH="${sorted_candidates[${#sorted_candidates[@]}-1]}"
else
  if ! [[ "${CHECKPOINT_EPOCH}" =~ ^[0-9]+$ ]] || (( CHECKPOINT_EPOCH < 1 )); then
    echo "ERROR: CHECKPOINT_EPOCH must be 'latest' or an integer >= 1, got '${CHECKPOINT_EPOCH}'" >&2
    exit 6
  fi
  CKPT_OBJECT_PATH="${RUN_DIR}/${RUN_NAME}_epoch_${CHECKPOINT_EPOCH}_object.ckpt"
fi

if [[ ! -f "${CKPT_OBJECT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_OBJECT_PATH}" >&2
  echo "Available checkpoints in ${RUN_DIR}:" >&2
  ls -1 "${RUN_DIR}"/*_object.ckpt >&2 || true
  exit 7
fi

if [[ "${CKPT_OBJECT_PATH}" != "${STABLEWM_HOME}/"* ]]; then
  echo "ERROR: checkpoint is not under STABLEWM_HOME; cannot derive policy path." >&2
  echo "STABLEWM_HOME=${STABLEWM_HOME}" >&2
  echo "CKPT_OBJECT_PATH=${CKPT_OBJECT_PATH}" >&2
  exit 8
fi

POLICY="${CKPT_OBJECT_PATH#${STABLEWM_HOME}/}"
POLICY="${POLICY%_object.ckpt}"
POLICY_BASENAME="$(basename "${POLICY}")"
RESULT_FILENAME="${RESULT_FILENAME:-${POLICY_BASENAME}_hi_pusht_results_d${GOAL_OFFSET_STEPS}.txt}"
RESULT_PATH="$(dirname "${CKPT_OBJECT_PATH}")/${RESULT_FILENAME}"

echo "Repo root: ${REPO_ROOT}"
echo "STABLEWM_HOME: ${STABLEWM_HOME}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint selection: ${CHECKPOINT_EPOCH}"
echo "Checkpoint object: ${CKPT_OBJECT_PATH}"
echo "Policy arg for hi_eval.py: ${POLICY}"
echo "Config name: ${CONFIG_NAME}"
echo "Goal offset steps (d): ${GOAL_OFFSET_STEPS}"
echo "Result file: ${RESULT_PATH}"

cd "${REPO_ROOT}"

# Compatibility for object checkpoints pickled from baseline code:
# torch.load may need top-level imports like `module` / `utils` from third_party/lewm.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"
fi
echo "PYTHONPATH prefix: ${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"

CMD=(
  python hi_eval.py
  --config-name="${CONFIG_NAME}"
  "policy=${POLICY}"
  "eval.goal_offset_steps=${GOAL_OFFSET_STEPS}"
  "output.filename=${RESULT_FILENAME}"
)

echo ""
echo "==> Launching eval command:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo ""
echo "Eval finished."
echo "Results appended to: ${RESULT_PATH}"
