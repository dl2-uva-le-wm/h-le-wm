#!/bin/bash

# Snellius CPU diagnostic job:
# Compare true dataset macro-actions vs CEM macro-actions for one-step high-level prediction,
# and measure off-manifold behavior of CEM macro-actions.
#
# Usage:
#   cd jobs/eval/hi
#   sbatch macro_action_manifold_cpu.sh
#
# Common overrides:
#   sbatch --export=ALL,CHECKPOINT_EPOCH=10 macro_action_manifold_cpu.sh
#   sbatch --export=ALL,RUN_NAME=hi_lewm_p2_train_hope1_21983875 macro_action_manifold_cpu.sh
#   sbatch --export=ALL,CHUNK_LEN_TOKENS=5,NUM_EVAL_SAMPLES=512 macro_action_manifold_cpu.sh
#   sbatch --export=ALL,CEM_SAMPLES=1500,CEM_ITERS=40 macro_action_manifold_cpu.sh
#   sbatch --export=ALL,CEM_BOUND_MODE=q01_q99 macro_action_manifold_cpu.sh
#   sbatch --export=ALL,DATASET_NAME=pusht_expert_train macro_action_manifold_cpu.sh

#SBATCH --partition=rome
#SBATCH --gpus=0
#SBATCH --job-name=macro_diag_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=macro_action_manifold_cpu_%j.out
#SBATCH --error=macro_action_manifold_cpu_%j.err

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
        if [[ -f "${p}/scripts/test_macro_action_manifold.py" && -f "${p}/hi_eval.py" ]]; then
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
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-latest}"  # latest or integer >= 1
DATASET_NAME="${DATASET_NAME:-pusht_expert_train}"
IMG_SIZE="${IMG_SIZE:-224}"
CHUNK_LEN_TOKENS="${CHUNK_LEN_TOKENS:-5}"
NUM_EVAL_SAMPLES="${NUM_EVAL_SAMPLES:-256}"
NUM_EMPIRICAL_CHUNKS="${NUM_EMPIRICAL_CHUNKS:-4096}"
CEM_SAMPLES="${CEM_SAMPLES:-900}"
CEM_ITERS="${CEM_ITERS:-20}"
CEM_ELITE_FRAC="${CEM_ELITE_FRAC:-0.1}"
CEM_BOUND_MODE="${CEM_BOUND_MODE:-none}"  # none | q01_q99 | q05_q95
SEED="${SEED:-42}"

RUN_DIR="${STABLEWM_HOME}/runs/${RUN_NAME}"

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
    echo "ERROR: CHECKPOINT_EPOCH must be latest or an integer >= 1, got '${CHECKPOINT_EPOCH}'" >&2
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

OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-macro_action_diag_job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
OUTPUT_DIR="$(dirname "${CKPT_OBJECT_PATH}")/${OUTPUT_SUBDIR}"
JSON_NAME="${JSON_NAME:-${POLICY_BASENAME}_macro_action_diag.json}"
JSON_PATH="${OUTPUT_DIR}/${JSON_NAME}"

echo "Repo root: ${REPO_ROOT}"
echo "STABLEWM_HOME: ${STABLEWM_HOME}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint selection: ${CHECKPOINT_EPOCH}"
echo "Checkpoint object: ${CKPT_OBJECT_PATH}"
echo "Policy arg: ${POLICY}"
echo "Dataset: ${DATASET_NAME}"
echo "chunk_len_tokens: ${CHUNK_LEN_TOKENS}"
echo "num_eval_samples: ${NUM_EVAL_SAMPLES}"
echo "num_empirical_chunks: ${NUM_EMPIRICAL_CHUNKS}"
echo "CEM: samples=${CEM_SAMPLES}, iters=${CEM_ITERS}, elite_frac=${CEM_ELITE_FRAC}, bound_mode=${CEM_BOUND_MODE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "JSON summary path: ${JSON_PATH}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

# Compatibility for object checkpoints pickled from baseline code:
# torch.load may need top-level imports like `module` / `utils` from third_party/lewm.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"
fi
echo "PYTHONPATH prefix: ${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"

# Force CPU behavior.
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

CMD=(
  python scripts/test_macro_action_manifold.py
  --policy "${POLICY}"
  --dataset-name "${DATASET_NAME}"
  --img-size "${IMG_SIZE}"
  --chunk-len-tokens "${CHUNK_LEN_TOKENS}"
  --num-eval-samples "${NUM_EVAL_SAMPLES}"
  --num-empirical-chunks "${NUM_EMPIRICAL_CHUNKS}"
  --cem-samples "${CEM_SAMPLES}"
  --cem-iters "${CEM_ITERS}"
  --cem-elite-frac "${CEM_ELITE_FRAC}"
  --cem-bound-mode "${CEM_BOUND_MODE}"
  --seed "${SEED}"
  --device cpu
  --save-json "${JSON_PATH}"
)

echo ""
echo "==> Launching diagnostic command:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo ""
echo "Diagnostic finished."
echo "JSON summary: ${JSON_PATH}"

