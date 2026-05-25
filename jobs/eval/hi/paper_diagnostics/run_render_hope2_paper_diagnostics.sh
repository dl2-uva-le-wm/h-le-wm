#!/usr/bin/env bash

#SBATCH --partition=fat_genoa
#SBATCH --job-name=hi_paper_render
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --chdir=/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics
#SBATCH --output=output/run_render_hope2_paper_diagnostics_%j.out
#SBATCH --error=output/run_render_hope2_paper_diagnostics_%j.err

set -euo pipefail

resolve_repo_root() {
  local c p
  for c in     "${PROJECT_ROOT:-}"     "${SLURM_SUBMIT_DIR:-}"     "${PWD:-}"     "${HOME}/main"     "/gpfs/home2/${USER}/main"; do
    [[ -z "${c}" ]] && continue
    for p in "${c}" "${c}/.." "${c}/../.." "${c}/../../.."; do
      if p="$(cd "${p}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${p}/scripts/diagnostics/render_hi_paper_diagnostics.py" ]]; then
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
  exit 2
fi
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_LOG_ROOT="${OFFLINE_LOG_ROOT:-${SCRIPT_DIR}/logs/offline}"
ACTING_LOG_ROOT="${ACTING_LOG_ROOT:-${SCRIPT_DIR}/logs/acting}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch-shared/${USER}/stablewm_data/reports/hi_paper_diagnostics_hope2_${SLURM_JOB_ID:-manual}}"
MATRIX_CSV="${MATRIX_CSV:-${REPO_ROOT}/roadmap/results/hi_eval_hope_matrix_status_2026-05-21.csv}"
BASELINE_MD="${BASELINE_MD:-${REPO_ROOT}/jobs/eval/original/pusht/baseline_matrix_results_2026-05-21.md}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
python scripts/diagnostics/render_hi_paper_diagnostics.py   --offline-log-root "${OFFLINE_LOG_ROOT}"   --acting-log-root "${ACTING_LOG_ROOT}"   --output-dir "${OUTPUT_DIR}"   --matrix-csv "${MATRIX_CSV}"   --baseline-md "${BASELINE_MD}"

echo "Report directory: ${OUTPUT_DIR}"
