#!/bin/bash

# CPU Rome array sweep for the resumed PushT VQ high-level checkpoint.
#
# Targets:
# - run: hi_lewm_p2_vq_train_hope1_22607223
# - checkpoint: epoch 20 (post-resume final checkpoint)
#
# Sweep design:
# - Focus on d=50, budget=50 to match the repo's medium-horizon PushT evals.
# - Keep LOW_ACTION_BLOCK=5 fixed because this checkpoint family expects grouped
#   low-level actions with that width.
# - Center the sweep on the established d50 H=2 regime.
# - Cover the main d50 variants already used elsewhere in the repo:
#   H=3 control, paper-scaled H=1/H=2, searchboost, and replan-3.
#
# Usage:
#   cd jobs/eval/hi/d50
#   sbatch vq_ep20_cpu_rome_array.sh
#
# Optional overrides:
#   sbatch --array=0-2 vq_ep20_cpu_rome_array.sh
#   sbatch --export=ALL,CHECKPOINT_EPOCH=latest vq_ep20_cpu_rome_array.sh
#   sbatch --export=ALL,RUN_NAME=hi_lewm_p2_vq_train_hope1_22607223 vq_ep20_cpu_rome_array.sh

#SBATCH --partition=rome
#SBATCH --gpus=0
#SBATCH --job-name=vqep20_d50
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=vq_ep20_cpu_rome_%A_%a.out
#SBATCH --error=vq_ep20_cpu_rome_%A_%a.err

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
  exit 2
fi

BASE_SCRIPT="${REPO_ROOT}/jobs/eval/hi/hope2/hope2_pusht_eval_base.sh"
if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: Could not locate hope2_pusht_eval_base.sh" >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

export RUN_NAME="${RUN_NAME:-hi_lewm_p2_vq_train_hope1_22607223}"
export CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-20}"
export EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
export GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-50}"
export EVAL_BUDGET="${EVAL_BUDGET:-50}"

export HIGH_RECEDING_HORIZON="${HIGH_RECEDING_HORIZON:-1}"
export HIGH_ACTION_BLOCK="${HIGH_ACTION_BLOCK:-1}"
export LOW_RECEDING_HORIZON="${LOW_RECEDING_HORIZON:-1}"
export LOW_ACTION_BLOCK="${LOW_ACTION_BLOCK:-5}"

case "${TASK_ID}" in
  0)
    LABEL="base_h2"
    HIGH_NUM_SAMPLES=1200
    HIGH_N_STEPS=30
    HIGH_TOPK=10
    HIGH_HORIZON=2
    HIGH_REPLAN_INTERVAL=5
    LOW_NUM_SAMPLES=300
    LOW_N_STEPS=30
    LOW_TOPK=150
    LOW_HORIZON=2
    ;;
  1)
    LABEL="h3_control"
    HIGH_NUM_SAMPLES=1200
    HIGH_N_STEPS=30
    HIGH_TOPK=10
    HIGH_HORIZON=3
    HIGH_REPLAN_INTERVAL=5
    LOW_NUM_SAMPLES=300
    LOW_N_STEPS=30
    LOW_TOPK=150
    LOW_HORIZON=2
    ;;
  2)
    LABEL="paperscaled_h1"
    HIGH_NUM_SAMPLES=1500
    HIGH_N_STEPS=40
    HIGH_TOPK=10
    HIGH_HORIZON=1
    HIGH_REPLAN_INTERVAL=5
    LOW_NUM_SAMPLES=900
    LOW_N_STEPS=20
    LOW_TOPK=150
    LOW_HORIZON=2
    ;;
  3)
    LABEL="paperscaled_h2"
    HIGH_NUM_SAMPLES=1500
    HIGH_N_STEPS=40
    HIGH_TOPK=10
    HIGH_HORIZON=2
    HIGH_REPLAN_INTERVAL=5
    LOW_NUM_SAMPLES=900
    LOW_N_STEPS=20
    LOW_TOPK=150
    LOW_HORIZON=2
    ;;
  4)
    LABEL="searchboost_h2"
    HIGH_NUM_SAMPLES=1500
    HIGH_N_STEPS=40
    HIGH_TOPK=20
    HIGH_HORIZON=2
    HIGH_REPLAN_INTERVAL=5
    LOW_NUM_SAMPLES=900
    LOW_N_STEPS=40
    LOW_TOPK=200
    LOW_HORIZON=2
    ;;
  5)
    LABEL="replan3_h2"
    HIGH_NUM_SAMPLES=1500
    HIGH_N_STEPS=40
    HIGH_TOPK=20
    HIGH_HORIZON=2
    HIGH_REPLAN_INTERVAL=3
    LOW_NUM_SAMPLES=900
    LOW_N_STEPS=40
    LOW_TOPK=200
    LOW_HORIZON=2
    ;;
  *)
    echo "ERROR: unsupported SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
    exit 2
    ;;
esac

export HIGH_NUM_SAMPLES="${HIGH_NUM_SAMPLES}"
export HIGH_N_STEPS="${HIGH_N_STEPS}"
export HIGH_TOPK="${HIGH_TOPK}"
export HIGH_HORIZON="${HIGH_HORIZON}"
export HIGH_REPLAN_INTERVAL="${HIGH_REPLAN_INTERVAL}"
export LOW_NUM_SAMPLES="${LOW_NUM_SAMPLES}"
export LOW_N_STEPS="${LOW_N_STEPS}"
export LOW_TOPK="${LOW_TOPK}"
export LOW_HORIZON="${LOW_HORIZON}"

export MODEL_LABEL="${MODEL_LABEL:-vq_ep20_${LABEL}}"
export EVAL_SUBDIR="${EVAL_SUBDIR:-eval_${MODEL_LABEL}_d50_b${EVAL_BUDGET}_job_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}}"
export RESULT_FILENAME="${RESULT_FILENAME:-${MODEL_LABEL}_results.txt}"

echo "Sweep label: ${LABEL}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint epoch: ${CHECKPOINT_EPOCH}"
echo "Goal offset: ${GOAL_OFFSET_STEPS}"
echo "Eval budget: ${EVAL_BUDGET}"
echo "High planner: h=${HIGH_HORIZON}, topk=${HIGH_TOPK}, samples=${HIGH_NUM_SAMPLES}, iters=${HIGH_N_STEPS}, replan=${HIGH_REPLAN_INTERVAL}"
echo "Low planner: h=${LOW_HORIZON}, topk=${LOW_TOPK}, samples=${LOW_NUM_SAMPLES}, iters=${LOW_N_STEPS}, block=${LOW_ACTION_BLOCK}"
echo "Eval subdir: ${EVAL_SUBDIR}"
echo "Result filename: ${RESULT_FILENAME}"

exec bash "${BASE_SCRIPT}"
