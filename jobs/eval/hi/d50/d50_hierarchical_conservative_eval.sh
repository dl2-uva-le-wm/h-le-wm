#!/bin/bash

# Conservative d=50 hierarchical PushT eval.
#
# Purpose:
# - Test whether d50 failure is mainly caused by speculative multi-step
#   high-level planning.
# - Keep the low-level setup close to the best d25 family while forcing the
#   high level to remain purely local.
#
# Defaults:
# - HIGH_HORIZON=1
# - LOW_HORIZON=2
# - LOW_TOPK kept high
# - HIGH_REPLAN_INTERVAL exposed for quick comparison (default: 3)
#
# Usage:
#   cd jobs/eval/hi/d50
#   sbatch d50_hierarchical_conservative_eval.sh
#
# Common overrides:
#   sbatch --export=ALL,HIGH_REPLAN_INTERVAL=5 d50_hierarchical_conservative_eval.sh
#   sbatch --export=ALL,CHECKPOINT_EPOCH=10 d50_hierarchical_conservative_eval.sh
#   sbatch --export=ALL,EVAL_DEVICE=cpu --partition=rome --gpus=0 d50_hierarchical_conservative_eval.sh

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=hi_eval_d50_hier_conservative
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=d50_hierarchical_conservative_eval_%j.out
#SBATCH --error=d50_hierarchical_conservative_eval_%j.err

set -euo pipefail

print_video_group() {
  local manifest_path="$1"
  local status_col="$2"
  local wanted_status="$3"
  local title="$4"

  echo "${title}"
  awk -F'\t' -v status_col="${status_col}" -v wanted_status="${wanted_status}" '
    NR == 1 {
      for (i = 1; i <= NF; i++) cols[$i] = i
      next
    }
    !(status_col in cols) { next }
    $cols[status_col] == wanted_status {
      found = 1
      printf "  [eval %s] %s (episode_id=%s, start_step=%s)\n",
        $cols["eval_index"], $cols["video_path"], $cols["episode_id"], $cols["start_step"]
    }
    END {
      if (!found) {
        printf "  none\n"
      }
    }
  ' "${manifest_path}"
}

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
  echo "Submit from repo root or pass PROJECT_ROOT=/path/to/h-lewm" >&2
  exit 2
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

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
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-latest}"
CONFIG_NAME="${CONFIG_NAME:-hi_pusht}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-50}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"

# Conservative hierarchical settings.
HIGH_NUM_SAMPLES="${HIGH_NUM_SAMPLES:-1200}"
HIGH_N_STEPS="${HIGH_N_STEPS:-30}"
HIGH_TOPK="${HIGH_TOPK:-10}"
HIGH_HORIZON="${HIGH_HORIZON:-1}"
HIGH_RECEDING_HORIZON="${HIGH_RECEDING_HORIZON:-1}"
HIGH_ACTION_BLOCK="${HIGH_ACTION_BLOCK:-1}"
HIGH_REPLAN_INTERVAL="${HIGH_REPLAN_INTERVAL:-3}"

LOW_NUM_SAMPLES="${LOW_NUM_SAMPLES:-300}"
LOW_N_STEPS="${LOW_N_STEPS:-30}"
LOW_TOPK="${LOW_TOPK:-150}"
LOW_HORIZON="${LOW_HORIZON:-2}"
LOW_RECEDING_HORIZON="${LOW_RECEDING_HORIZON:-1}"
LOW_ACTION_BLOCK="${LOW_ACTION_BLOCK:-5}"

EVAL_SUBDIR="${EVAL_SUBDIR:-eval_hier_conservative_hh1_lh2_k${HIGH_REPLAN_INTERVAL}_d${GOAL_OFFSET_STEPS}_job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
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
RESULT_FILENAME="${RESULT_FILENAME:-${POLICY_BASENAME}_hi_pusht_results_d${GOAL_OFFSET_STEPS}_hier_conservative.txt}"
ARTIFACTS_DIR="$(dirname "${CKPT_OBJECT_PATH}")/${EVAL_SUBDIR}"
RESULT_PATH="${ARTIFACTS_DIR}/${RESULT_FILENAME}"
MANIFEST_PATH="${ARTIFACTS_DIR}/${RESULT_FILENAME%.*}_episodes.tsv"

echo "Repo root: ${REPO_ROOT}"
echo "STABLEWM_HOME: ${STABLEWM_HOME}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint selection: ${CHECKPOINT_EPOCH}"
echo "Checkpoint object: ${CKPT_OBJECT_PATH}"
echo "Policy arg for hi_eval.py: ${POLICY}"
echo "Config name: ${CONFIG_NAME}"
echo "Goal offset steps (d): ${GOAL_OFFSET_STEPS}"
echo "Eval budget: ${EVAL_BUDGET}"
echo "Eval device: ${EVAL_DEVICE}"
echo "Planning mode: hierarchical"
echo "High-level planner: horizon=${HIGH_HORIZON}, receding=${HIGH_RECEDING_HORIZON}, block=${HIGH_ACTION_BLOCK}, samples=${HIGH_NUM_SAMPLES}, iters=${HIGH_N_STEPS}, topk=${HIGH_TOPK}, k=${HIGH_REPLAN_INTERVAL}"
echo "Low-level planner: horizon=${LOW_HORIZON}, receding=${LOW_RECEDING_HORIZON}, block=${LOW_ACTION_BLOCK}, samples=${LOW_NUM_SAMPLES}, iters=${LOW_N_STEPS}, topk=${LOW_TOPK}"
echo "Output subdir: ${EVAL_SUBDIR}"
echo "Artifacts dir: ${ARTIFACTS_DIR}"
echo "Result file: ${RESULT_PATH}"
echo "Episode manifest: ${MANIFEST_PATH}"

cd "${REPO_ROOT}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"
fi
echo "PYTHONPATH prefix: ${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"

if [[ "${EVAL_DEVICE}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
fi

CMD=(
  python hi_eval.py
  --config-name="${CONFIG_NAME}"
  "device=${EVAL_DEVICE}"
  "policy=${POLICY}"
  "planning.mode=hierarchical"
  "eval.goal_offset_steps=${GOAL_OFFSET_STEPS}"
  "eval.eval_budget=${EVAL_BUDGET}"
  "output.subdir=${EVAL_SUBDIR}"
  "planning.high.replan_interval=${HIGH_REPLAN_INTERVAL}"
  "planning.high.solver.device=${EVAL_DEVICE}"
  "planning.high.solver.num_samples=${HIGH_NUM_SAMPLES}"
  "planning.high.solver.n_steps=${HIGH_N_STEPS}"
  "planning.high.solver.topk=${HIGH_TOPK}"
  "planning.high.plan_config.horizon=${HIGH_HORIZON}"
  "planning.high.plan_config.receding_horizon=${HIGH_RECEDING_HORIZON}"
  "planning.high.plan_config.action_block=${HIGH_ACTION_BLOCK}"
  "planning.low.solver.device=${EVAL_DEVICE}"
  "planning.low.solver.num_samples=${LOW_NUM_SAMPLES}"
  "planning.low.solver.n_steps=${LOW_N_STEPS}"
  "planning.low.solver.topk=${LOW_TOPK}"
  "planning.low.plan_config.horizon=${LOW_HORIZON}"
  "planning.low.plan_config.receding_horizon=${LOW_RECEDING_HORIZON}"
  "planning.low.plan_config.action_block=${LOW_ACTION_BLOCK}"
  "solver.device=${EVAL_DEVICE}"
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
echo "Artifacts written to: ${ARTIFACTS_DIR}"
echo "Results appended to: ${RESULT_PATH}"
if [[ -f "${MANIFEST_PATH}" ]]; then
  echo "Episode manifest written to: ${MANIFEST_PATH}"
  echo ""
  echo "==> Video outcomes"
  print_video_group "${MANIFEST_PATH}" "status" "PASS" "Passing videos (env metric):"
  print_video_group "${MANIFEST_PATH}" "status" "FAIL" "Failing videos (env metric):"
  print_video_group "${MANIFEST_PATH}" "status_block_only" "PASS" "Passing videos (block-only metric):"
  print_video_group "${MANIFEST_PATH}" "status_block_only" "FAIL" "Failing videos (block-only metric):"
else
  echo "WARNING: Episode manifest not found: ${MANIFEST_PATH}" >&2
fi
