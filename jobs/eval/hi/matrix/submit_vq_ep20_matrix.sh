#!/bin/bash

# Submit the fixed-stride matrix for the resumed PushT VQ checkpoint.
#
# Target checkpoint:
# - run: hi_lewm_p2_vq_train_hope1_22607223
# - epoch: 20
#
# Usage:
#   cd jobs/eval/hi/matrix
#   ./submit_vq_ep20_matrix.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT_FILE="${CHECKPOINT_FILE:-${SCRIPT_DIR}/checkpoints_vq_ep20.txt}"
export JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval_fixed_stride_matrix.sh}"

exec bash "${SCRIPT_DIR}/submit_fixed_stride_matrix.sh"
