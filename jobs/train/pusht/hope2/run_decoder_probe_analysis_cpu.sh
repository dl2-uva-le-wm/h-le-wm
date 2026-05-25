#!/usr/bin/env bash

# CPU-only decoder-probe analysis/export job.
# Reproduces the notebook workflow headlessly and writes a structured report
# folder with metrics, galleries, rollout stories, and comparison figures.
#
# Usage:
#   cd jobs/train/pusht
#   sbatch hope2/run_decoder_probe_analysis_cpu.sh
#
# Optional overrides:
#   PHASE_B_EPOCH=10 sbatch hope2/run_decoder_probe_analysis_cpu.sh
#   OUTPUT_DIR=/scratch-shared/$USER/stablewm_data/reports/my_probe_report sbatch hope2/run_decoder_probe_analysis_cpu.sh

#SBATCH --partition=fat_genoa
#SBATCH --job-name=hi_decoder_probe_report
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=output/hope2/run_decoder_probe_report_cpu_%j.out
#SBATCH --error=output/hope2/run_decoder_probe_report_cpu_%j.err

set -euo pipefail

resolve_repo_root() {
  local c p
  for c in     "${PROJECT_ROOT:-}"     "${SLURM_SUBMIT_DIR:-}"     "${PWD:-}"     "${HOME}/main"     "${HOME}/h-le-wm"     "${HOME}/h-lewm"     "/gpfs/home2/${USER}/main"     "/gpfs/home2/${USER}/h-le-wm"     "/gpfs/home2/${USER}/h-lewm"; do
    [[ -z "${c}" ]] && continue
    for p in "${c}" "${c}/.." "${c}/../.." "${c}/../../.."; do
      if p="$(cd "${p}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${p}/scripts/diagnostics/run_decoder_probe_report.py" && -f "${p}/scripts/diagnostics/decoder_probe_notebook_utils.py" ]]; then
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

BASELINE_ROOT="${REPO_ROOT}/third_party/lewm"
if [[ ! -f "${BASELINE_ROOT}/module.py" || ! -f "${BASELINE_ROOT}/utils.py" ]]; then
  echo "ERROR: Baseline submodule is missing or incomplete at ${BASELINE_ROOT}." >&2
  exit 2
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

set +u
eval "$(conda shell.bash hook)"
conda activate lewm-gpu
set -u

mkdir -p "${SLURM_SUBMIT_DIR:-${REPO_ROOT}/jobs/train/pusht}/output/hope2"

SHARED_CACHE="${SHARED_CACHE:-/scratch-shared/${USER}/stablewm_data}"
RUNS_ROOT="${RUNS_ROOT:-${SHARED_CACHE}/runs}"
REPORTS_ROOT="${REPORTS_ROOT:-${SHARED_CACHE}/reports}"
PHASE_A_RUN="${PHASE_A_RUN:-}"
PHASE_A_PREFIX="${PHASE_A_PREFIX:-hi_decoder_probe_true_hope2_}"
PHASE_B_EPOCH="${PHASE_B_EPOCH:-10}"
BATCH_SIZE="${ANALYSIS_BATCH_SIZE:-16}"
NUM_WORKERS="${ANALYSIS_NUM_WORKERS:-8}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-12}"
TRAIN_MAX_BATCHES="${TRAIN_MAX_BATCHES:-6}"
VAL_GALLERY_EXAMPLES="${VAL_GALLERY_EXAMPLES:-12}"
TRAIN_GALLERY_EXAMPLES="${TRAIN_GALLERY_EXAMPLES:-6}"
COMPARISON_SAMPLES="${COMPARISON_SAMPLES:-2}"
EPOCH_COMPARE="${EPOCH_COMPARE:-1,5,10}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPORTS_ROOT}/decoder_probe_analysis_cpu_${SLURM_JOB_ID}}"

CACHE_DIR="${SHARED_CACHE}"
if [[ -n "${TMPDIR:-}" && -d "${TMPDIR}" ]]; then
  LOCAL_CACHE="${TMPDIR}/stablewm_data"
  mkdir -p "${LOCAL_CACHE}"
  export MPLCONFIGDIR="${TMPDIR}/matplotlib-${SLURM_JOB_ID}"
  mkdir -p "${MPLCONFIGDIR}"
  rsync -a "${SHARED_CACHE}/pusht_expert_train.h5" "${LOCAL_CACHE}/"
  if [[ -f "${SHARED_CACHE}/pusht_expert_val.h5" ]]; then
    rsync -a "${SHARED_CACHE}/pusht_expert_val.h5" "${LOCAL_CACHE}/"
  fi
  CACHE_DIR="${LOCAL_CACHE}"
fi

mkdir -p "${OUTPUT_DIR}"

PHASE_B_RUN="${PHASE_B_RUN:-}"
PHASE_B_PREFIX="${PHASE_B_PREFIX:-hi_decoder_probe_pred_exposed_hope2_}"
if [[ -z "${PHASE_A_RUN}" ]]; then
  PHASE_A_RUN="$(python - <<PY2
from pathlib import Path
runs_root = Path('${RUNS_ROOT}')
candidates = sorted([p for p in runs_root.glob('${PHASE_A_PREFIX}*') if p.is_dir()], key=lambda p: p.stat().st_mtime)
print(candidates[-1] if candidates else '')
PY2
)"
fi
if [[ -z "${PHASE_B_RUN}" ]]; then
  PHASE_B_RUN="$(python - <<PY2
from pathlib import Path
runs_root = Path('${RUNS_ROOT}')
candidates = sorted([p for p in runs_root.glob('${PHASE_B_PREFIX}*') if p.is_dir()], key=lambda p: p.stat().st_mtime)
print(candidates[-1] if candidates else '')
PY2
)"
fi

CMD=(
  python scripts/diagnostics/run_decoder_probe_report.py
  --output-dir "${OUTPUT_DIR}"
  --cache-dir "${CACHE_DIR}"
  --phase-b-epoch "${PHASE_B_EPOCH}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --val-max-batches "${VAL_MAX_BATCHES}"
  --train-max-batches "${TRAIN_MAX_BATCHES}"
  --val-gallery-examples "${VAL_GALLERY_EXAMPLES}"
  --train-gallery-examples "${TRAIN_GALLERY_EXAMPLES}"
  --comparison-samples "${COMPARISON_SAMPLES}"
  --epoch-compare "${EPOCH_COMPARE}"
  --device cpu
)
if [[ -n "${PHASE_A_RUN}" ]]; then
  CMD+=(--phase-a-run "${PHASE_A_RUN}")
fi
if [[ -n "${PHASE_B_RUN}" ]]; then
  CMD+=(--phase-b-run "${PHASE_B_RUN}")
fi

cd "${REPO_ROOT}"
"${CMD[@]}"

printf '
Report folder: %s
' "${OUTPUT_DIR}"
if [[ -f "${OUTPUT_DIR}/report.md" ]]; then
  printf '
===== report.md =====
'
  sed -n '1,120p' "${OUTPUT_DIR}/report.md"
fi
