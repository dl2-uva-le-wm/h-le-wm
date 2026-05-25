#!/usr/bin/env bash

#SBATCH --partition=rome
#SBATCH --gpus=0
#SBATCH --job-name=hi_dec_story_compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --chdir=/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics
#SBATCH --output=output/run_compute_hope2_decoder_story_artifacts_%j.out
#SBATCH --error=output/run_compute_hope2_decoder_story_artifacts_%j.err

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
        if [[ -f "${p}/scripts/diagnostics/run_hi_acting_diagnostic.py" ]]; then
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

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
RUN_NAME="${RUN_NAME:-hi_lewm_p2_train_hope2_22253175}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-15}"
POLICY="runs/${RUN_NAME}/${RUN_NAME}_epoch_${CHECKPOINT_EPOCH}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch-shared/${USER}/stablewm_data/reports/hi_decoder_story_artifacts_hope2_${SLURM_JOB_ID:-manual}}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_DIR}/mplconfig}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OUTPUT_DIR}/xdg-cache}"
mkdir -p "${OUTPUT_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}/third_party/lewm:${REPO_ROOT}"
fi

COMMON_ARGS=(
  --policy "${POLICY}"
  --eval-config "${REPO_ROOT}/h_le_wm/config/eval/hi_pusht.yaml"
  --cache-dir "${STABLEWM_HOME}"
  --goal-offset-steps 50
  --eval-budget 50
  --num-eval 50
  --high-num-samples 1500
  --high-iters 40
  --high-topk 10
  --low-num-samples 900
  --low-iters 20
  --low-topk 150
  --frame-skip 5
  --num-reference-samples 4096
  --seed 42
  --device cpu
)

cd "${REPO_ROOT}"

python scripts/diagnostics/run_hi_acting_diagnostic.py \
  "${COMMON_ARGS[@]}" \
  --experiment-kind generated_subgoal_acting \
  --high-horizon 2 \
  --low-horizon 2 \
  --low-receding-horizon 1 \
  --save-json "${OUTPUT_DIR}/generated_subgoal_acting_d50_hh2_lh2_lrh1.json" \
  --save-npz "${OUTPUT_DIR}/generated_subgoal_acting_d50_hh2_lh2_lrh1.npz" \
  --append-tsv "${OUTPUT_DIR}/summary_generated_subgoal_acting.tsv"

python scripts/diagnostics/run_hi_acting_diagnostic.py \
  "${COMMON_ARGS[@]}" \
  --experiment-kind online_hierarchical_logging \
  --high-horizon 2 \
  --low-horizon 2 \
  --low-receding-horizon 1 \
  --save-json "${OUTPUT_DIR}/online_hierarchical_logging_d50_hh2_lh2_lrh1.json" \
  --save-npz "${OUTPUT_DIR}/online_hierarchical_logging_d50_hh2_lh2_lrh1.npz" \
  --append-tsv "${OUTPUT_DIR}/summary_online_hierarchical_logging.tsv"

echo "Decoder story artifacts: ${OUTPUT_DIR}"
