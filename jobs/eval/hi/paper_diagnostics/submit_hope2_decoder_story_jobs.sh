#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE_WORKER="${COMPUTE_WORKER:-${SCRIPT_DIR}/run_compute_hope2_decoder_story_artifacts.sh}"
RENDER_WORKER="${RENDER_WORKER:-${SCRIPT_DIR}/run_render_hope2_decoder_story_figures.sh}"

if [[ ! -f "${COMPUTE_WORKER}" ]]; then
  echo "ERROR: compute worker not found: ${COMPUTE_WORKER}" >&2
  exit 2
fi
if [[ ! -f "${RENDER_WORKER}" ]]; then
  echo "ERROR: render worker not found: ${RENDER_WORKER}" >&2
  exit 2
fi

compute_job_id="$(sbatch --parsable "${COMPUTE_WORKER}")"
artifact_dir="/scratch-shared/${USER}/stablewm_data/reports/hi_decoder_story_artifacts_hope2_${compute_job_id}"
generated_artifact="${artifact_dir}/generated_subgoal_acting_d50_hh2_lh2_lrh1.npz"
online_artifact="${artifact_dir}/online_hierarchical_logging_d50_hh2_lh2_lrh1.npz"

render_job_id="$(
  sbatch --parsable \
    --dependency="afterok:${compute_job_id}" \
    --export="ALL,GENERATED_ARTIFACT=${generated_artifact},ONLINE_ARTIFACT=${online_artifact}" \
    "${RENDER_WORKER}"
)"

cat <<EOF
compute_job_id=${compute_job_id}
render_job_id=${render_job_id}
artifact_dir=${artifact_dir}
generated_artifact=${generated_artifact}
online_artifact=${online_artifact}
EOF
