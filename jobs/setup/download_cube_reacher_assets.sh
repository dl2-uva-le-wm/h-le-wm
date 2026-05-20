#!/bin/bash

# Snellius job: download the official Cube/Reacher datasets and convert the
# official Hugging Face base LeWM checkpoints into object checkpoints under
# shared scratch, if they are not already present.
#
# Usage:
#   sbatch jobs/setup/download_cube_reacher_assets.sh
# Optional overrides:
#   sbatch --export=ALL,STABLEWM_HOME=/scratch-shared/$USER/stablewm_data jobs/setup/download_cube_reacher_assets.sh
#   sbatch --export=ALL,TARGETS=cube jobs/setup/download_cube_reacher_assets.sh

#SBATCH --partition=rome
#SBATCH --job-name=DownloadCubeReacher
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00

set -euo pipefail

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
        if [[ -f "${p}/scripts/setup_datasets.sh" && -f "${p}/scripts/convert_hf_weights_to_object_ckpt.py" ]]; then
          echo "${p}"
          return 0
        fi
      fi
    done
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root with dataset/model setup helpers." >&2
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

LOG_DIR="${REPO_ROOT}/jobs/setup/out"
mkdir -p "${LOG_DIR}"
JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%s)}"
exec > >(tee -a "${LOG_DIR}/download_cube_reacher_assets_${JOB_TAG}.out") \
     2> >(tee -a "${LOG_DIR}/download_cube_reacher_assets_${JOB_TAG}.err" >&2)

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
TARGETS="${TARGETS:-cube,reacher}"

CUBE_HF_URL="${CUBE_HF_URL:-https://huggingface.co/quentinll/lewm-cube/tree/main}"
REACHER_HF_URL="${REACHER_HF_URL:-https://huggingface.co/quentinll/lewm-reacher/tree/main}"
CUBE_RUN_NAME="${CUBE_RUN_NAME:-cube/lewm}"
REACHER_RUN_NAME="${REACHER_RUN_NAME:-reacher/lewm}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "STABLEWM_HOME=${STABLEWM_HOME}"
echo "TARGETS=${TARGETS}"
echo "CUBE_HF_URL=${CUBE_HF_URL}"
echo "REACHER_HF_URL=${REACHER_HF_URL}"
echo "CUBE_RUN_NAME=${CUBE_RUN_NAME}"
echo "REACHER_RUN_NAME=${REACHER_RUN_NAME}"

mkdir -p "${STABLEWM_HOME}"

cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/setup_datasets.sh" \
  --home "${STABLEWM_HOME}" \
  --datasets "${TARGETS}"

download_model_if_missing() {
  local target="$1"
  local hf_url run_name ckpt_path dataset_path

  case "${target}" in
    cube)
      hf_url="${CUBE_HF_URL}"
      run_name="${CUBE_RUN_NAME}"
      ckpt_path="${STABLEWM_HOME}/${run_name}_object.ckpt"
      dataset_path="${STABLEWM_HOME}/ogbench/cube_single_expert.h5"
      ;;
    reacher)
      hf_url="${REACHER_HF_URL}"
      run_name="${REACHER_RUN_NAME}"
      ckpt_path="${STABLEWM_HOME}/${run_name}_object.ckpt"
      dataset_path="${STABLEWM_HOME}/dmc/reacher_random.h5"
      ;;
    *)
      echo "ERROR: Unsupported target '${target}'. Use cube,reacher." >&2
      exit 3
      ;;
  esac

  if [[ ! -f "${dataset_path}" ]]; then
    echo "ERROR: expected dataset is still missing after setup: ${dataset_path}" >&2
    exit 4
  fi

  if [[ -f "${ckpt_path}" ]]; then
    echo "[skip] checkpoint already exists: ${ckpt_path}"
    return 0
  fi

  echo "[download] model ${target} from ${hf_url}"
  python scripts/convert_hf_weights_to_object_ckpt.py \
    --hf-url "${hf_url}" \
    --run-name "${run_name}"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "ERROR: expected checkpoint missing after conversion: ${ckpt_path}" >&2
    exit 5
  fi

  echo "[ok] checkpoint saved: ${ckpt_path}"
}

IFS=',' read -r -a TARGET_LIST <<< "${TARGETS}"
for raw_target in "${TARGET_LIST[@]}"; do
  target="$(echo "${raw_target}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  [[ -z "${target}" ]] && continue
  download_model_if_missing "${target}"
done

echo ""
echo "Asset setup completed."
echo "Cube dataset: ${STABLEWM_HOME}/ogbench/cube_single_expert.h5"
echo "Cube checkpoint: ${STABLEWM_HOME}/${CUBE_RUN_NAME}_object.ckpt"
echo "Reacher dataset: ${STABLEWM_HOME}/dmc/reacher_random.h5"
echo "Reacher checkpoint: ${STABLEWM_HOME}/${REACHER_RUN_NAME}_object.ckpt"
