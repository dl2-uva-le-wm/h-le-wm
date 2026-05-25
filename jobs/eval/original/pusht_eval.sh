#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=orig_eval_pusht
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=out/pusht_eval_%j.out
#SBATCH --error=out/pusht_eval_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
COMMON_HELPER="${REPO_ROOT}/jobs/eval/common/determinism_env.sh"
mkdir -p "${SCRIPT_DIR}/out"

if [[ ! -f "${COMMON_HELPER}" ]]; then
  echo "ERROR: determinism helper not found: ${COMMON_HELPER}" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${COMMON_HELPER}"

module purge
module load 2025
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
conda activate lewm-gpu

cd "${REPO_ROOT}"

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"

setup_eval_determinism_env   "${REPO_ROOT}"   "${EVAL_SEED:-42}"   "${EVAL_DETERMINISM:-strict}"

if [[ "${EVAL_DEVICE}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

print_eval_determinism_env

python third_party/lewm/eval.py   --config-name=pusht.yaml   policy=pusht/lewm   "seed=${EVAL_SEED}"   "+eval.device=${EVAL_DEVICE}"   "solver.device=${EVAL_DEVICE}"
