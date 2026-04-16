#!/bin/bash

# Snellius job: set STABLEWM_HOME and download the TwoRooms dataset.
# Usage:
#   sbatch jobs/setup/download_tworooms.sh
# Optional override:
#   sbatch --export=ALL,STABLEWM_HOME=/scratch-shared/$USER/stablewm_data jobs/setup/download_tworooms.sh

#SBATCH --partition=rome
#SBATCH --job-name=DownloadTwoRooms
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --chdir=jobs/setup
#SBATCH --output=out/download_tworooms_%j.out
#SBATCH --error=out/download_tworooms_%j.err

set -eo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
if conda env list | grep -E '(^|[[:space:]])lewm([[:space:]]|$)' >/dev/null 2>&1; then
  conda activate lewm
fi

resolve_repo_root() {
  local c
  for c in "${PROJECT_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${HOME}/h-le-wm"; do
    if [[ -n "${c}" && -f "${c}/scripts/setup_datasets.sh" ]]; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root with scripts/setup_datasets.sh" >&2
  echo "Checked: PROJECT_ROOT='${PROJECT_ROOT:-}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', HOME/h-le-wm='${HOME}/h-le-wm'" >&2
  exit 2
fi

LOG_DIR="${REPO_ROOT}/jobs/setup/out"
mkdir -p "${LOG_DIR}"
JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%s)}"
exec > >(tee -a "${LOG_DIR}/download_tworooms_${JOB_TAG}.out") \
     2> >(tee -a "${LOG_DIR}/download_tworooms_${JOB_TAG}.err" >&2)

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
DATASETS="tworooms"

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
