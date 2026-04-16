#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00

set -eo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

resolve_repo_root() {
  local c p
  for c in \
    "${PROJECT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${HOME}/h-lewm" \
    "${HOME}/h-le-wm" \
    "/gpfs/home2/${USER}/h-lewm" \
    "/gpfs/home2/${USER}/h-le-wm"; do
    [[ -z "${c}" ]] && continue
    for p in "${c}" "${c}/.." "${c}/../.."; do
      if p="$(cd "${p}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${p}/environment-gpu.yml" ]]; then
          echo "${p}"
          return 0
        fi
      fi
    done
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root with environment-gpu.yml" >&2
  echo "Checked: PROJECT_ROOT='${PROJECT_ROOT:-}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', PWD='${PWD:-}', HOME='${HOME:-}'" >&2
  exit 2
fi

LOG_DIR="${REPO_ROOT}/jobs/setup/out"
mkdir -p "${LOG_DIR}"
JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%s)}"
exec > >(tee -a "${LOG_DIR}/install_env_${JOB_TAG}.out") \
     2> >(tee -a "${LOG_DIR}/install_env_${JOB_TAG}.err" >&2)

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

cd "${REPO_ROOT}"

conda env create -f environment-gpu.yml
