#!/bin/bash

# CPU diagnostic job: solve the high-level plan once, keep the full staged
# rollout, and let only the low-level planner replan under each fixed stage
# target.
#
# Usage:
#   sbatch --export=ALL,CHECKPOINT_NAME=hi_lewm_p2_train_latent_action_dim_32_stride_5_n4_22569364_epoch_15 jobs/tests/staged_high_plan_cpu.sh
#
# Common overrides:
#   sbatch --export=ALL,CHECKPOINT_NAME=...,GOAL_OFFSET_STEPS=50 staged_high_plan_cpu.sh
#   sbatch --export=ALL,CHECKPOINT_NAME=...,EVAL_DEVICE=cuda staged_high_plan_cpu.sh
#   sbatch --export=ALL,CHECKPOINT_NAME=...,STAGE_DURATION_STEPS=25 staged_high_plan_cpu.sh
#   sbatch --export=ALL,CHECKPOINT_NAME=...,HIGH_NUM_SAMPLES=1500,LOW_NUM_SAMPLES=900 staged_high_plan_cpu.sh

#SBATCH --partition=rome
#SBATCH --gpus=0
#SBATCH --job-name=hi_stage_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=staged_high_plan_cpu_%j.out
#SBATCH --error=staged_high_plan_cpu_%j.err

set -euo pipefail

resolve_repo_root() {
  local c p
  for c in     "${PROJECT_ROOT:-}"     "${SLURM_SUBMIT_DIR:-}"     "${PWD:-}"     "${HOME}/main"     "${HOME}/h-le-wm"     "${HOME}/h-lewm"     "/gpfs/home2/${USER}/main"     "/gpfs/home2/${USER}/h-le-wm"     "/gpfs/home2/${USER}/h-lewm"; do
    [[ -z "${c}" ]] && continue
    for p in "${c}" "${c}/.." "${c}/../.."; do
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
  echo "Submit from repo root or pass PROJECT_ROOT=/path/to/main" >&2
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

CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"
if [[ -z "${CHECKPOINT_NAME}" ]]; then
  echo "ERROR: CHECKPOINT_NAME is required." >&2
  echo "Example: sbatch --export=ALL,CHECKPOINT_NAME=hi_lewm_p2_train_latent_action_dim_32_stride_5_n4_22569364_epoch_15 jobs/tests/staged_high_plan_cpu.sh" >&2
  exit 3
fi

normalize_checkpoint_name() {
  local name="$1"
  name="${name%_object.ckpt}"
  name="${name%.ckpt}"
  echo "${name}"
}

CHECKPOINT_BASE="$(normalize_checkpoint_name "${CHECKPOINT_NAME}")"
if [[ ! "${CHECKPOINT_BASE}" =~ ^(.+)_epoch_([0-9]+)$ ]]; then
  echo "ERROR: CHECKPOINT_NAME must look like <run_name>_epoch_<N> (optionally with .ckpt or _object.ckpt)." >&2
  echo "Got: ${CHECKPOINT_NAME}" >&2
  exit 4
fi
RUN_NAME="${BASH_REMATCH[1]}"
CHECKPOINT_EPOCH="${BASH_REMATCH[2]}"

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
RUN_DIR="${STABLEWM_HOME}/runs/${RUN_NAME}"
CKPT_OBJECT_PATH="${RUN_DIR}/${CHECKPOINT_BASE}_object.ckpt"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run directory not found: ${RUN_DIR}" >&2
  exit 5
fi

if [[ ! -f "${CKPT_OBJECT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_OBJECT_PATH}" >&2
  echo "Available object checkpoints in ${RUN_DIR}:" >&2
  ls -1 "${RUN_DIR}"/*_object.ckpt >&2 || true
  exit 6
fi

if [[ "${CKPT_OBJECT_PATH}" != "${STABLEWM_HOME}/"* ]]; then
  echo "ERROR: checkpoint is not under STABLEWM_HOME; cannot derive policy path." >&2
  echo "STABLEWM_HOME=${STABLEWM_HOME}" >&2
  echo "CKPT_OBJECT_PATH=${CKPT_OBJECT_PATH}" >&2
  exit 7
fi

POLICY="${CKPT_OBJECT_PATH#${STABLEWM_HOME}/}"
POLICY="${POLICY%_object.ckpt}"
POLICY_BASENAME="$(basename "${POLICY}")"

CONFIG_NAME="${CONFIG_NAME:-hi_pusht}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-50}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"

HIGH_NUM_SAMPLES="${HIGH_NUM_SAMPLES:-1500}"
HIGH_N_STEPS="${HIGH_N_STEPS:-40}"
HIGH_TOPK="${HIGH_TOPK:-10}"
HIGH_HORIZON="${HIGH_HORIZON:-2}"
HIGH_RECEDING_HORIZON="${HIGH_RECEDING_HORIZON:-1}"
HIGH_ACTION_BLOCK="${HIGH_ACTION_BLOCK:-1}"
HIGH_REPLAN_INTERVAL="${HIGH_REPLAN_INTERVAL:-5}"

LOW_NUM_SAMPLES="${LOW_NUM_SAMPLES:-900}"
LOW_N_STEPS="${LOW_N_STEPS:-20}"
LOW_TOPK="${LOW_TOPK:-150}"
LOW_HORIZON="${LOW_HORIZON:-2}"
LOW_RECEDING_HORIZON="${LOW_RECEDING_HORIZON:-1}"
LOW_ACTION_BLOCK="${LOW_ACTION_BLOCK:-5}"

OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-eval_hier_staged_d${GOAL_OFFSET_STEPS}_job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
RESULT_FILENAME="${RESULT_FILENAME:-${POLICY_BASENAME}_hi_pusht_results_d${GOAL_OFFSET_STEPS}_staged.txt}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"
fi

if [[ "${EVAL_DEVICE}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

echo "Repo root: ${REPO_ROOT}"
echo "STABLEWM_HOME: ${STABLEWM_HOME}"
echo "Checkpoint name: ${CHECKPOINT_BASE}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint epoch: ${CHECKPOINT_EPOCH}"
echo "Checkpoint object: ${CKPT_OBJECT_PATH}"
echo "Policy arg for hi_eval.py: ${POLICY}"
echo "Config name: ${CONFIG_NAME}"
echo "Goal offset steps (d): ${GOAL_OFFSET_STEPS}"
echo "Eval budget: ${EVAL_BUDGET}"
echo "Eval device: ${EVAL_DEVICE}"
echo "Planning mode: hierarchical_staged"
echo "High-level planner: horizon=${HIGH_HORIZON}, receding=${HIGH_RECEDING_HORIZON}, block=${HIGH_ACTION_BLOCK}, samples=${HIGH_NUM_SAMPLES}, iters=${HIGH_N_STEPS}, topk=${HIGH_TOPK}"
echo "Low-level planner: horizon=${LOW_HORIZON}, receding=${LOW_RECEDING_HORIZON}, block=${LOW_ACTION_BLOCK}, samples=${LOW_NUM_SAMPLES}, iters=${LOW_N_STEPS}, topk=${LOW_TOPK}"
if [[ -n "${STAGE_DURATION_STEPS:-}" ]]; then
  echo "Stage duration override: ${STAGE_DURATION_STEPS} env steps"
else
  echo "Stage duration override: auto (derived inside hi_eval.py)"
fi
echo "Output subdir: ${OUTPUT_SUBDIR}"

echo
cd "${REPO_ROOT}"

CMD=(
  python hi_eval.py
  --config-name="${CONFIG_NAME}"
  "policy=${POLICY}"
  "planning.mode=hierarchical_staged"
  "eval.goal_offset_steps=${GOAL_OFFSET_STEPS}"
  "eval.eval_budget=${EVAL_BUDGET}"
  "output.subdir=${OUTPUT_SUBDIR}"
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

if [[ -n "${STAGE_DURATION_STEPS:-}" ]]; then
  CMD+=("+planning.staged.stage_duration_steps=${STAGE_DURATION_STEPS}")
fi
if [[ -n "${CLEAR_LOW_BUFFER_ON_STAGE_CHANGE:-}" ]]; then
  CMD+=("+planning.staged.clear_low_buffer_on_stage_change=${CLEAR_LOW_BUFFER_ON_STAGE_CHANGE}")
fi

echo "==> Launching staged diagnostic command:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo
echo "Staged diagnostic finished."
