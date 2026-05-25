#!/usr/bin/env bash

#SBATCH --partition=fat_genoa
#SBATCH --job-name=hi_dec_story_render
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --chdir=/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics
#SBATCH --output=output/run_render_hope2_decoder_story_figures_%j.out
#SBATCH --error=output/run_render_hope2_decoder_story_figures_%j.err

set -euo pipefail

resolve_repo_root() {
  local c p
  for c in \
    "${PROJECT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${HOME}/main" \
    "/gpfs/home2/${USER}/main"; do
    [[ -z "${c}" ]] && continue
    for p in "${c}" "${c}/.." "${c}/../.." "${c}/../../.."; do
      if p="$(cd "${p}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${p}/scripts/render_hi_decoder_diagnostic_stories.py" ]]; then
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

PROBE_RUN_DIR="${PROBE_RUN_DIR:-/scratch-shared/${USER}/stablewm_data/runs/hi_decoder_probe_pred_exposed_hope2_20260522_121648}"
TEACHER_ARTIFACT="${TEACHER_ARTIFACT:-/scratch-shared/${USER}/stablewm_data/runs/hi_lewm_p2_train_hope2_22253175/diag_teacher_vs_open_loop_d50_hh2_lh2_job_23048413_8/teacher_vs_open_loop_d50_hh2_lh2.npz}"
GENERATED_ARTIFACT="${GENERATED_ARTIFACT:-}"
ONLINE_ARTIFACT="${ONLINE_ARTIFACT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch-shared/${USER}/stablewm_data/reports/hi_decoder_story_figures_hope2_${SLURM_JOB_ID:-manual}}"
STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_DIR}/mplconfig}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OUTPUT_DIR}/xdg-cache}"
mkdir -p "${OUTPUT_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

if [[ -z "${GENERATED_ARTIFACT}" || -z "${ONLINE_ARTIFACT}" ]]; then
  echo "ERROR: GENERATED_ARTIFACT and ONLINE_ARTIFACT must be set." >&2
  exit 3
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"
fi

cd "${REPO_ROOT}"
python scripts/render_hi_decoder_diagnostic_stories.py \
  --probe-run-dir "${PROBE_RUN_DIR}" \
  --teacher-artifact "${TEACHER_ARTIFACT}" \
  --generated-artifact "${GENERATED_ARTIFACT}" \
  --online-artifact "${ONLINE_ARTIFACT}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${STABLEWM_HOME}" \
  --goal-offset-steps 50 \
  --frame-skip 5

echo "Decoder story figures: ${OUTPUT_DIR}"
