#!/bin/bash

# Submit ablation training jobs for H-LeWM PushT.
#
# Ablation matrix:
#   fix123  — config-only fixes: latent_dim=32, stride=1, 50ep (no code changes needed for eval)
#   full    — all fixes: above + depth=8, preembed=True, lambda_var=0.1
#
# After training, run eval with:
#   sbatch --export=ALL,POLICY=hi_abl_fix123_epoch_50,LOW_HORIZON=2 jobs/eval/hi/pusht_eval_v2.sh
#   sbatch --export=ALL,POLICY=hi_abl_full_epoch_50,LOW_HORIZON=2   jobs/eval/hi/pusht_eval_v2.sh
#
# Also test RC-1 on existing baseline checkpoint (no retraining):
#   sbatch --export=ALL,POLICY=<existing_ckpt>,LOW_HORIZON=2 jobs/eval/hi/pusht_eval_v2.sh
#   sbatch --export=ALL,POLICY=<existing_ckpt>,LOW_HORIZON=5 jobs/eval/hi/pusht_eval_v2.sh
#
# Usage:
#   cd jobs/2_levels/pusht
#   bash ablation_suite.sh

set -euo pipefail

CKPT="${PRETRAINED_LEWM_CKPT:-/scratch-shared/${USER}/stablewm_data/pusht/lewm_object.ckpt}"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: pretrained LeWM checkpoint not found: ${CKPT}" >&2
  echo "Set PRETRAINED_LEWM_CKPT=/path/to/lewm_object.ckpt" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Submitting ablation training jobs ==="
echo "Pretrained checkpoint: ${CKPT}"
echo ""

# fix123: config-only (dim=32, stride=1, 50ep, baseline capacity, no preembed, no var_reg)
JID1=$(sbatch \
  --job-name=abl_fix123 \
  --export=ALL,\
RUN_NAME=hi_abl_fix123,\
LATENT_DIM=32,\
PREEMBED=False,\
LAMBDA_VAR=0.0,\
STRIDE=1,\
DEPTH_HIGH=6,\
MLP_HIGH=2048,\
MAX_EPOCHS=50,\
PRETRAINED_LEWM_CKPT="${CKPT}" \
  "${SCRIPT_DIR}/train_v2.sh" | awk '{print $NF}')
echo "  fix123 (RC-2+3+4 only)  → SLURM job ${JID1}"

# full: all changes (dim=32, stride=1, 50ep, depth=8, preembed=True, var_reg=0.1)
JID2=$(sbatch \
  --job-name=abl_full \
  --export=ALL,\
RUN_NAME=hi_abl_full,\
LATENT_DIM=32,\
PREEMBED=True,\
LAMBDA_VAR=0.1,\
STRIDE=1,\
DEPTH_HIGH=8,\
MLP_HIGH=4096,\
MAX_EPOCHS=50,\
PRETRAINED_LEWM_CKPT="${CKPT}" \
  "${SCRIPT_DIR}/train_v2.sh" | awk '{print $NF}')
echo "  full (all RC fixes)     → SLURM job ${JID2}"

echo ""
echo "=== Eval commands (run after training completes) ==="
echo ""
echo "  # RC-1 immediate test on existing checkpoint (no retraining)"
echo "  LOW_HORIZON=2 sbatch --export=ALL,POLICY=<existing_ckpt> jobs/eval/hi/pusht_eval_v2.sh"
echo "  LOW_HORIZON=5 sbatch --export=ALL,POLICY=<existing_ckpt> jobs/eval/hi/pusht_eval_v2.sh"
echo ""
echo "  # fix123 ablation"
echo "  sbatch --export=ALL,POLICY=hi_abl_fix123_epoch_50,LOW_HORIZON=2 jobs/eval/hi/pusht_eval_v2.sh"
echo "  sbatch --export=ALL,POLICY=hi_abl_fix123_epoch_50,LOW_HORIZON=1 jobs/eval/hi/pusht_eval_v2.sh"
echo ""
echo "  # full ablation"
echo "  sbatch --export=ALL,POLICY=hi_abl_full_epoch_50,LOW_HORIZON=2 jobs/eval/hi/pusht_eval_v2.sh"
echo "  sbatch --export=ALL,POLICY=hi_abl_full_epoch_50,LOW_HORIZON=1 jobs/eval/hi/pusht_eval_v2.sh"
