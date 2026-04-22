#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWEEPS_ROOT="${SCRIPT_DIR}/sweeps"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_NAME="overnight_cpu_rome_${STAMP}"
SWEEP_DIR="${SWEEPS_ROOT}/${SWEEP_NAME}"

mkdir -p "${SWEEP_DIR}"

PAPER_SCRIPT="${SCRIPT_DIR}/d50_hierarchical_soft_low_h2_paper_scaled_eval.sh"
SOFT_SCRIPT="${SCRIPT_DIR}/d50_hierarchical_soft_low_h2_eval.sh"
DEFAULT_SCRIPT="${SCRIPT_DIR}/d50_hierarchical_default_eval.sh"

for f in "${PAPER_SCRIPT}" "${SOFT_SCRIPT}" "${DEFAULT_SCRIPT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing script ${f}" >&2
    exit 2
  fi
done

COMMON_SBATCH_ARGS=(
  --parsable
  --partition=rome
  --gpus=0
  --cpus-per-task=8
  --time=12:00:00
)

JOBS_TSV="${SWEEP_DIR}/submitted_jobs.tsv"
LOG_TXT="${SWEEP_DIR}/submit.log"

echo -e "job_id\tlabel\tscript\teval_subdir\tslurm_out\tslurm_err" > "${JOBS_TSV}"

submit_job() {
  local label="$1"
  local script_path="$2"
  local extra_exports="$3"

  local eval_subdir="${SWEEP_NAME}/${label}"
  local slurm_out="${SWEEP_DIR}/${label}_%j.out"
  local slurm_err="${SWEEP_DIR}/${label}_%j.err"

  local export_vars="ALL,CHECKPOINT_EPOCH=10,EVAL_DEVICE=cpu,GOAL_OFFSET_STEPS=50,EVAL_BUDGET=50,EVAL_SUBDIR=${eval_subdir}"
  if [[ -n "${extra_exports}" ]]; then
    export_vars+="${extra_exports}"
  fi

  local job_id
  job_id=$(sbatch "${COMMON_SBATCH_ARGS[@]}" \
    --job-name="d50_${label}" \
    -o "${slurm_out}" \
    -e "${slurm_err}" \
    --export="${export_vars}" \
    "${script_path}")

  echo "[$(date '+%F %T')] submitted ${label}: ${job_id}" | tee -a "${LOG_TXT}"
  echo -e "${job_id}\t${label}\t${script_path}\t${eval_subdir}\t${slurm_out}\t${slurm_err}" >> "${JOBS_TSV}"
}

submit_job "a_baseline_pscaled" "${PAPER_SCRIPT}" ",HIGH_HORIZON=1,HIGH_TOPK=10,HIGH_REPLAN_INTERVAL=5,HIGH_NUM_SAMPLES=1500,HIGH_N_STEPS=40,LOW_HORIZON=2,LOW_TOPK=150,LOW_NUM_SAMPLES=900,LOW_N_STEPS=20"
submit_job "b_highh2_topk30" "${PAPER_SCRIPT}" ",HIGH_HORIZON=2,HIGH_TOPK=30,HIGH_REPLAN_INTERVAL=5,HIGH_NUM_SAMPLES=1500,HIGH_N_STEPS=40,LOW_HORIZON=2,LOW_TOPK=150,LOW_NUM_SAMPLES=900,LOW_N_STEPS=20"
submit_job "c_highh2_topk30_replan3" "${PAPER_SCRIPT}" ",HIGH_HORIZON=2,HIGH_TOPK=30,HIGH_REPLAN_INTERVAL=3,HIGH_NUM_SAMPLES=1500,HIGH_N_STEPS=40,LOW_HORIZON=2,LOW_TOPK=150,LOW_NUM_SAMPLES=900,LOW_N_STEPS=20"
submit_job "d_highh2_replan3_lowsteps30_topk90" "${PAPER_SCRIPT}" ",HIGH_HORIZON=2,HIGH_TOPK=30,HIGH_REPLAN_INTERVAL=3,HIGH_NUM_SAMPLES=1500,HIGH_N_STEPS=40,LOW_HORIZON=2,LOW_TOPK=90,LOW_NUM_SAMPLES=900,LOW_N_STEPS=30"
submit_job "e_soft_lowh2_default" "${SOFT_SCRIPT}" ""
submit_job "f_d50_default_control" "${DEFAULT_SCRIPT}" ""

echo "Sweep folder: ${SWEEP_DIR}" | tee -a "${LOG_TXT}"
echo "Jobs manifest: ${JOBS_TSV}" | tee -a "${LOG_TXT}"
