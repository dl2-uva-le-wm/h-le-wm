#!/bin/bash
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
mkdir -p "${SCRIPT_DIR}/out"

module purge
module load 2025
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
conda activate lewm-gpu

# Submit from: ~/h-le-wm/jobs/eval/original
cd ../../..

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"

python third_party/lewm/eval.py --config-name=pusht.yaml policy=pusht/lewm
