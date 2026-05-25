#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-${SCRIPT_DIR}/checkpoints_hope2.txt}"
OFFLINE_WORKER="${OFFLINE_WORKER:-${REPO_ROOT}/jobs/eval/hi/diagnostics/run_diagnostics_matrix.sh}"
ACTING_WORKER="${ACTING_WORKER:-${REPO_ROOT}/jobs/eval/hi/acting_diagnostics/run_acting_diagnostics_matrix.sh}"
RENDER_WORKER="${RENDER_WORKER:-${SCRIPT_DIR}/run_render_hope2_paper_diagnostics.sh}"
OFFLINE_LOG_ROOT="${OFFLINE_LOG_ROOT:-${SCRIPT_DIR}/logs/offline}"
ACTING_LOG_ROOT="${ACTING_LOG_ROOT:-${SCRIPT_DIR}/logs/acting}"
DECODER_DEPENDENCY_JOB_ID="${DECODER_DEPENDENCY_JOB_ID:-}"
mkdir -p "${OFFLINE_LOG_ROOT}/hi_lewm_p2_train_hope2_22253175" "${ACTING_LOG_ROOT}/hi_lewm_p2_train_hope2_22253175"

OFFLINE_JOB_ID="$(sbatch --parsable   --array=1-20   --output="${OFFLINE_LOG_ROOT}/hi_lewm_p2_train_hope2_22253175/run_diagnostics_matrix_%A_%a.out"   --error="${OFFLINE_LOG_ROOT}/hi_lewm_p2_train_hope2_22253175/run_diagnostics_matrix_%A_%a.err"   --export="ALL,CHECKPOINT_ROW_INDEX=1,JOB_SCRIPT_DIR=${REPO_ROOT}/jobs/eval/hi/diagnostics,CHECKPOINT_FILE=${CHECKPOINT_FILE},LOG_ROOT=${OFFLINE_LOG_ROOT}"   "${OFFLINE_WORKER}")"

echo "Submitted offline diagnostics array: ${OFFLINE_JOB_ID}"

ACTING_JOB_ID="$(sbatch --parsable   --array=1-12   --output="${ACTING_LOG_ROOT}/hi_lewm_p2_train_hope2_22253175/run_acting_diagnostics_matrix_%A_%a.out"   --error="${ACTING_LOG_ROOT}/hi_lewm_p2_train_hope2_22253175/run_acting_diagnostics_matrix_%A_%a.err"   --export="ALL,CHECKPOINT_ROW_INDEX=1,JOB_SCRIPT_DIR=${REPO_ROOT}/jobs/eval/hi/acting_diagnostics,CHECKPOINT_FILE=${CHECKPOINT_FILE},LOG_ROOT=${ACTING_LOG_ROOT}"   "${ACTING_WORKER}")"

echo "Submitted acting diagnostics array: ${ACTING_JOB_ID}"

RENDER_DEPENDENCY="afterok:${OFFLINE_JOB_ID}:${ACTING_JOB_ID}"
if [[ -n "${DECODER_DEPENDENCY_JOB_ID}" ]]; then
  RENDER_DEPENDENCY="${RENDER_DEPENDENCY}:${DECODER_DEPENDENCY_JOB_ID}"
fi
RENDER_JOB_ID="$(sbatch --parsable   --dependency="${RENDER_DEPENDENCY}"   --export="ALL,OFFLINE_LOG_ROOT=${OFFLINE_LOG_ROOT},ACTING_LOG_ROOT=${ACTING_LOG_ROOT}"   "${RENDER_WORKER}")"

echo "Submitted render job: ${RENDER_JOB_ID}"

echo "offline_job_id=${OFFLINE_JOB_ID}"
echo "acting_job_id=${ACTING_JOB_ID}"
echo "render_job_id=${RENDER_JOB_ID}"
