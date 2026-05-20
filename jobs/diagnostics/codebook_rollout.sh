#!/bin/bash
#
# Codebook rollout diagnostic: one rollout per VQ codebook entry.
# High-level CEM replaced by fixed codebook lookup; low-level CEM kept.
#
# Usage (submit from repo root or any directory):
#   sbatch --export=ALL,CHECKPOINT_NAME=<run>_epoch_<N> jobs/diagnostics/codebook_rollout.sh
#
# Optional env overrides:
#   N_FRAMES=50          total env steps per codebook action (default: 50)
#   NUM_ACTIONS=128      how many codebook actions to render; unset = all
#   NUM_CODES=128        deprecated alias for NUM_ACTIONS
#   START_STEP_SEED=0    seed for picking the starting state from the dataset
#   LOW_NUM_SAMPLES=600  low-level CEM samples (reduce for faster/cheaper runs)
#   LOW_N_STEPS=30       low-level CEM iterations
#   LOW_TOPK=60          low-level CEM top-k
#   EVAL_DEVICE=cuda     or "cpu"; use cpu + MUJOCO_GL=osmesa for rome partition
#   OUTPUT_SUBDIR=...    override the output subdirectory name
#
# To run on CPU partition (slower but no GPU quota):
#   Change --partition=rome, --gpus=0, and export EVAL_DEVICE=cpu
#   Also export LOW_NUM_SAMPLES=200 LOW_N_STEPS=10 to keep runtime under 4h.

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=codebook_rollout
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=out/codebook_rollout_%j.out
#SBATCH --error=out/codebook_rollout_%j.err

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
  echo "Set PROJECT_ROOT=/path/to/h-le-wm or submit from within the repo." >&2
  exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${REPO_ROOT}/jobs/diagnostics}"
mkdir -p "${SUBMIT_DIR}/out"

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
  echo "ERROR: conda env 'lewm-gpu' or 'lewm' not found." >&2
  echo "Run jobs/setup/setup_env.sh first, or create the env from environment-gpu.yml." >&2
  exit 2
fi
set -u

CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"
if [[ -z "${CHECKPOINT_NAME}" ]]; then
  echo "ERROR: CHECKPOINT_NAME is required." >&2
  echo "Example:" >&2
  echo "  sbatch --export=ALL,CHECKPOINT_NAME=hi_lewm_p2_train_latent_action_dim_32_stride_5_n4_22569364_epoch_15 \\" >&2
  echo "         jobs/diagnostics/codebook_rollout.sh" >&2
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
  ls -1 "${RUN_DIR}"/*_object.ckpt 2>/dev/null >&2 || echo "  (none found)" >&2
  exit 6
fi

POLICY="${CKPT_OBJECT_PATH#${STABLEWM_HOME}/}"
POLICY="${POLICY%_object.ckpt}"
POLICY_BASENAME="$(basename "${POLICY}")"

EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
N_FRAMES="${N_FRAMES:-50}"
START_STEP_SEED="${START_STEP_SEED:-0}"
LOW_NUM_SAMPLES="${LOW_NUM_SAMPLES:-600}"
LOW_N_STEPS="${LOW_N_STEPS:-30}"
LOW_TOPK="${LOW_TOPK:-60}"
REPLAN_INTERVAL="${REPLAN_INTERVAL:-5}"
NUM_ACTIONS="${NUM_ACTIONS:-}"
if [[ -n "${NUM_CODES:-}" ]]; then
  if [[ -n "${NUM_ACTIONS}" && "${NUM_ACTIONS}" != "${NUM_CODES}" ]]; then
    echo "ERROR: NUM_ACTIONS and deprecated NUM_CODES were both set differently." >&2
    echo "Use NUM_ACTIONS only." >&2
    exit 7
  fi
  NUM_ACTIONS="${NUM_CODES}"
fi
JOB_TOKEN="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-codebook_rollout_job_${JOB_TOKEN}}"
OUTPUT_BASE="${STABLEWM_HOME}/$(dirname "${POLICY}")/diagnostics"

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

echo "REPO_ROOT=${REPO_ROOT}"
echo "STABLEWM_HOME=${STABLEWM_HOME}"
echo "POLICY=${POLICY}"
echo "CHECKPOINT=${CKPT_OBJECT_PATH}"
echo "EVAL_DEVICE=${EVAL_DEVICE}"
echo "N_FRAMES=${N_FRAMES}"
echo "NUM_ACTIONS=${NUM_ACTIONS:-all}"
echo "START_STEP_SEED=${START_STEP_SEED}"
echo "LOW CEM: num_samples=${LOW_NUM_SAMPLES}, n_steps=${LOW_N_STEPS}, topk=${LOW_TOPK}"
echo "REPLAN_INTERVAL=${REPLAN_INTERVAL}"
echo "Output: ${OUTPUT_BASE}/${OUTPUT_SUBDIR}"

cd "${REPO_ROOT}"

CMD=(
  python scripts/codebook_rollout_experiment.py
  --config-name=codebook_rollout
  "policy=${POLICY}"
  "experiment.n_frames=${N_FRAMES}"
  "experiment.start_step_seed=${START_STEP_SEED}"
  "experiment.output_dir=${OUTPUT_BASE}/${OUTPUT_SUBDIR}"
  "planning.high.replan_interval=${REPLAN_INTERVAL}"
  "planning.low.solver.device=${EVAL_DEVICE}"
  "planning.low.solver.num_samples=${LOW_NUM_SAMPLES}"
  "planning.low.solver.n_steps=${LOW_N_STEPS}"
  "planning.low.solver.topk=${LOW_TOPK}"
)

# Optionally limit the number of codebook actions (useful for quick tests).
if [[ -n "${NUM_ACTIONS}" ]]; then
  CMD+=( "experiment.num_actions=${NUM_ACTIONS}" )
fi

echo "==> Launching codebook rollout:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo
echo "Codebook rollout finished."
echo "Videos and summary.json: ${OUTPUT_BASE}/${OUTPUT_SUBDIR}/"
echo "Canonical action videos: ${OUTPUT_BASE}/${OUTPUT_SUBDIR}/videos/action_<index>.mp4"
