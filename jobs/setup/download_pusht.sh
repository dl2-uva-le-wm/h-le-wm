#!/bin/bash

# Snellius job: set STABLEWM_HOME and download LeWM datasets (default: PushT).
# Usage:
#   sbatch jobs/setup/download_pusht.sh
# Optional overrides:
#   sbatch --export=ALL,DATASETS=pusht,tworooms jobs/setup/download_pusht.sh
#   sbatch --export=ALL,STABLEWM_HOME=/scratch-shared/$USER/stablewm_data jobs/setup/download_pusht.sh

#SBATCH --partition=rome
#SBATCH --job-name=DownloadLeWMDatasets
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=download_datasets_%j.out
#SBATCH --error=download_datasets_%j.err

set -eo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

# If the project environment exists, use it (helps ensure zstd/python deps exist).
eval "$(conda shell.bash hook)"
if conda env list | grep -E '(^|[[:space:]])lewm([[:space:]]|$)' >/dev/null 2>&1; then
  conda activate lewm
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
DATASETS="${DATASETS:-pusht}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "STABLEWM_HOME=${STABLEWM_HOME}"
echo "DATASETS=${DATASETS}"

mkdir -p "${STABLEWM_HOME}"

source "${REPO_ROOT}/scripts/setup_datasets.sh" \
  --home "${STABLEWM_HOME}" \
  --datasets "${DATASETS}"

echo ""
echo "Dataset job completed."
echo "Use this in future jobs:"
echo "  export STABLEWM_HOME=\"${STABLEWM_HOME}\""
