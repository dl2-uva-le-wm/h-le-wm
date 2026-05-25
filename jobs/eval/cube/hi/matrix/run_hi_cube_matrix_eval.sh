#!/bin/bash

# Shared Cube hierarchical eval launcher for matrix jobs.
#
# Environment variables are populated by eval_hope_hierarchical_matrix.sh.

set -euo pipefail

resolve_repo_root() {
  local c p
  for c in \
    "${PROJECT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${HOME}/main" \
    "/gpfs/home2/${USER}/main"; do
    [[ -z "${c}" ]] && continue
    for p in "${c}" "${c}/.." "${c}/../.." "${c}/../../.."; do
      if p="$(cd "${p}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${p}/h_le_wm/eval/hierarchical.py" && -f "${p}/h_le_wm/config/eval/hi_cube.yaml" ]]; then
          echo "${p}"
          return 0
        fi
      fi
    done
  done
  return 1
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required environment variable is unset: ${name}" >&2
    exit 2
  fi
}

append_override_if_set() {
  local env_name="$1"
  local hydra_key="$2"
  if [[ -n "${!env_name:-}" ]]; then
    CMD+=("${hydra_key}=${!env_name}")
  fi
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root." >&2
  exit 2
fi

COMMON_HELPER="${REPO_ROOT}/jobs/eval/common/determinism_env.sh"
if [[ ! -f "${COMMON_HELPER}" ]]; then
  echo "ERROR: determinism helper not found: ${COMMON_HELPER}" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${COMMON_HELPER}"

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
  exit 2
fi
set -u

for name in \
  RUN_NAME CHECKPOINT_EPOCH CONFIG_NAME PLANNING_MODE NUM_EVAL EVAL_DEVICE \
  GOAL_OFFSET_STEPS EVAL_BUDGET HIGH_HORIZON LOW_HORIZON HIGH_RECEDING_HORIZON \
  LOW_RECEDING_HORIZON HIGH_REPLAN_INTERVAL HIGH_ACTION_BLOCK LOW_ACTION_BLOCK \
  HIGH_NUM_SAMPLES HIGH_N_STEPS HIGH_TOPK LOW_NUM_SAMPLES LOW_N_STEPS LOW_TOPK \
  EVAL_SUBDIR RESULT_FILENAME; do
  require_env "${name}"
done

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
RUN_DIR="${STABLEWM_HOME}/runs/${RUN_NAME}"
CKPT_OBJECT_PATH="${RUN_DIR}/${RUN_NAME}_epoch_${CHECKPOINT_EPOCH}_object.ckpt"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run directory not found: ${RUN_DIR}" >&2
  exit 3
fi

if [[ ! -f "${CKPT_OBJECT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_OBJECT_PATH}" >&2
  exit 4
fi

POLICY="${CKPT_OBJECT_PATH#${STABLEWM_HOME}/}"
POLICY="${POLICY%_object.ckpt}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"
fi

setup_eval_determinism_env \
  "${REPO_ROOT}" \
  "${EVAL_SEED:-42}" \
  "${EVAL_DETERMINISM:-strict}"

if [[ "${EVAL_DEVICE}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

echo "Repo root: ${REPO_ROOT}"
echo "STABLEWM_HOME: ${STABLEWM_HOME}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint epoch: ${CHECKPOINT_EPOCH}"
echo "Checkpoint object: ${CKPT_OBJECT_PATH}"
echo "Policy arg for hierarchical eval: ${POLICY}"
echo "Config name: ${CONFIG_NAME}"
echo "Planning mode: ${PLANNING_MODE}"
echo "Goal offset steps: ${GOAL_OFFSET_STEPS}"
echo "Eval budget: ${EVAL_BUDGET}"
echo "Num eval: ${NUM_EVAL}"
echo "Eval device: ${EVAL_DEVICE}"
print_eval_determinism_env
echo "High-level planner: horizon=${HIGH_HORIZON}, receding=${HIGH_RECEDING_HORIZON}, block=${HIGH_ACTION_BLOCK}, samples=${HIGH_NUM_SAMPLES}, iters=${HIGH_N_STEPS}, topk=${HIGH_TOPK}, replan=${HIGH_REPLAN_INTERVAL}"
echo "Low-level planner: horizon=${LOW_HORIZON}, receding=${LOW_RECEDING_HORIZON}, block=${LOW_ACTION_BLOCK}, samples=${LOW_NUM_SAMPLES}, iters=${LOW_N_STEPS}, topk=${LOW_TOPK}"
echo "Output subdir: ${EVAL_SUBDIR}"
echo "Result filename: ${RESULT_FILENAME}"
if [[ -n "${EMPIRICAL_MACRO_ENABLED:-}" ]]; then
  echo "Empirical macro high solver: enabled=${EMPIRICAL_MACRO_ENABLED}, num_sequences=${EMPIRICAL_MACRO_NUM_SEQUENCES:-<config>}, chunk_len=${EMPIRICAL_MACRO_CHUNK_LEN:-<config>}, residual_scale=${EMPIRICAL_MACRO_RESIDUAL_SCALE:-<config>}, stage_sampling=${EMPIRICAL_MACRO_STAGE_SAMPLING:-<config>}"
fi

cd "${REPO_ROOT}"

CMD=(
  python -m h_le_wm.eval.hierarchical
  --config-name="${CONFIG_NAME}"
  "policy=${POLICY}"
  "seed=${EVAL_SEED}"
  "planning.mode=${PLANNING_MODE}"
  "eval.num_eval=${NUM_EVAL}"
  "eval.goal_offset_steps=${GOAL_OFFSET_STEPS}"
  "eval.eval_budget=${EVAL_BUDGET}"
  "output.subdir=${EVAL_SUBDIR}"
  "output.filename=${RESULT_FILENAME}"
  "planning.high.solver.device=${EVAL_DEVICE}"
  "planning.low.solver.device=${EVAL_DEVICE}"
  "solver.device=${EVAL_DEVICE}"
  "planning.high.solver.num_samples=${HIGH_NUM_SAMPLES}"
  "planning.high.solver.n_steps=${HIGH_N_STEPS}"
  "planning.high.solver.topk=${HIGH_TOPK}"
  "planning.high.plan_config.horizon=${HIGH_HORIZON}"
  "planning.high.plan_config.receding_horizon=${HIGH_RECEDING_HORIZON}"
  "planning.high.plan_config.action_block=${HIGH_ACTION_BLOCK}"
  "planning.high.replan_interval=${HIGH_REPLAN_INTERVAL}"
  "planning.low.solver.num_samples=${LOW_NUM_SAMPLES}"
  "planning.low.solver.n_steps=${LOW_N_STEPS}"
  "planning.low.solver.topk=${LOW_TOPK}"
  "planning.low.plan_config.horizon=${LOW_HORIZON}"
  "planning.low.plan_config.receding_horizon=${LOW_RECEDING_HORIZON}"
  "planning.low.plan_config.action_block=${LOW_ACTION_BLOCK}"
)

append_override_if_set EMPIRICAL_MACRO_ENABLED planning.high.empirical_macro.enabled
append_override_if_set EMPIRICAL_MACRO_NUM_SEQUENCES planning.high.empirical_macro.num_sequences
append_override_if_set EMPIRICAL_MACRO_CHUNK_LEN planning.high.empirical_macro.chunk_len
append_override_if_set EMPIRICAL_MACRO_RESIDUAL_SCALE planning.high.empirical_macro.residual_scale
append_override_if_set EMPIRICAL_MACRO_MIN_RESIDUAL_STD planning.high.empirical_macro.min_residual_std
append_override_if_set EMPIRICAL_MACRO_RETURN_TOP_CANDIDATES planning.high.empirical_macro.return_top_candidates
append_override_if_set EMPIRICAL_MACRO_ENCODE_BATCH_SIZE planning.high.empirical_macro.encode_batch_size
append_override_if_set EMPIRICAL_MACRO_STAGE_SAMPLING planning.high.empirical_macro.stage_sampling
append_override_if_set EMPIRICAL_MACRO_SEED planning.high.empirical_macro.seed

echo
echo "==> Launching matrix eval command:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo
echo "Matrix eval finished."
