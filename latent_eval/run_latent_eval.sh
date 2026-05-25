#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_latent_eval.sh
# Runs the latent trajectory evaluation for hi_lewm_p2_train_hope2 checkpoint.
#
# Usage:
#   cd latent_eval
#   bash run_latent_eval.sh [optional overrides]
#
# Optional environment variables:
#   CHECKPOINT       path to the .ckpt file (default: auto-detected)
#   DATASET_NAME     HDF5 dataset name       (default: pusht_expert_train)
#   CACHE_DIR        STABLEWM cache dir      (default: auto)
#   NUM_EVAL         number of trajectories  (default: 50)
#   GOAL_OFFSET      goal offset steps       (default: 25)
#   EVAL_BUDGET      steps per episode       (default: 50)
#   SEED             random seed             (default: 42)
#   DEVICE           auto | cuda | cpu       (default: auto)
#   OUTPUT_DIR       where to save results   (default: ./results)
#   HIGH_SAMPLES     CEM high-level samples  (default: 300)
#   LOW_SAMPLES      CEM low-level  samples  (default: 200)
#   RECORD_SUBGOALS  record subgoals csv     (default: 1)
#   SKIP_ANALYSIS    skip analyse_latents.py (default: 0)
#
# Examples:
#   bash run_latent_eval.sh
#   NUM_EVAL=5 DEVICE=cpu bash run_latent_eval.sh   # quick smoke test
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── locate repo root ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── defaults ──────────────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/hi_lewm_p2_train_hope2_22253175_weights.ckpt}"
DATASET_NAME="${DATASET_NAME:-pusht_expert_train}"
CACHE_DIR="${CACHE_DIR:-}"
NUM_EVAL="${NUM_EVAL:-50}"
GOAL_OFFSET="${GOAL_OFFSET:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
HIGH_SAMPLES="${HIGH_SAMPLES:-300}"
LOW_SAMPLES="${LOW_SAMPLES:-200}"
RECORD_SUBGOALS="${RECORD_SUBGOALS:-1}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"

# ── validate checkpoint ───────────────────────────────────────────────────────
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

echo "════════════════════════════════════════"
echo " Latent trajectory evaluation"
echo "════════════════════════════════════════"
echo " repo root    : ${REPO_ROOT}"
echo " checkpoint   : ${CHECKPOINT}"
echo " dataset      : ${DATASET_NAME}"
echo " cache dir    : ${CACHE_DIR:-<auto>}"
echo " num eval     : ${NUM_EVAL}"
echo " goal offset  : ${GOAL_OFFSET}"
echo " eval budget  : ${EVAL_BUDGET}"
echo " seed         : ${SEED}"
echo " device       : ${DEVICE}"
echo " output       : ${OUTPUT_DIR}"
echo " CEM high/low : ${HIGH_SAMPLES} / ${LOW_SAMPLES}"
echo " record subgoals: ${RECORD_SUBGOALS}"
echo "════════════════════════════════════════"
echo ""

# ── build python args ─────────────────────────────────────────────────────────
ARGS=(
    --checkpoint   "${CHECKPOINT}"
    --dataset-name "${DATASET_NAME}"
    --num-eval     "${NUM_EVAL}"
    --goal-offset  "${GOAL_OFFSET}"
    --eval-budget  "${EVAL_BUDGET}"
    --seed         "${SEED}"
    --device       "${DEVICE}"
    --output-dir   "${OUTPUT_DIR}"
    --high-num-samples "${HIGH_SAMPLES}"
    --low-num-samples  "${LOW_SAMPLES}"
)
if [[ -n "${CACHE_DIR}" ]]; then
    ARGS+=(--cache-dir "${CACHE_DIR}")
fi
if [[ "${RECORD_SUBGOALS}" == "1" ]]; then
    ARGS+=(--record-subgoals)
fi

# ── run ───────────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"

# Resolve python3: prefer the venv that has stable_pretraining installed.
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    for candidate in \
        "${REPO_ROOT}/.venv/bin/python3" \
        "$(dirname "${REPO_ROOT}")/old/le-wm/.venv/bin/python3" \
        "${HOME}/Desktop/uva/courses/dl2/old/le-wm/.venv/bin/python3"; do
        if [[ -x "${candidate}" ]] && "${candidate}" -c "import stable_pretraining" 2>/dev/null; then
            PYTHON="${candidate}"
            break
        fi
    done
fi
if [[ -z "${PYTHON}" ]]; then
    # Fall back to whatever python3 is on PATH (user may have activated the env)
    PYTHON="python3"
fi
echo "Python  : ${PYTHON}"

"${PYTHON}" latent_eval/latent_eval.py "${ARGS[@]}"

echo ""
echo "Done. Latents saved to: ${OUTPUT_DIR}"

# ── analysis ──────────────────────────────────────────────────────────────────
if [[ "${SKIP_ANALYSIS}" != "1" ]]; then
    echo ""
    echo "════════════════════════════════════════"
    echo " Running analyse_latents.py"
    echo "════════════════════════════════════════"

    ANALYSIS_ARGS=(
        --csv          "${OUTPUT_DIR}/latent_trajectories.csv"
        --output-dir   "${OUTPUT_DIR}/analysis"
        --replan-interval 5
        --dtw-pca-dim  20
        --frechet-pca-dim 20
    )
    if [[ "${RECORD_SUBGOALS}" == "1" ]]; then
        ANALYSIS_ARGS+=(--subgoal-csv "${OUTPUT_DIR}/subgoal_records.csv")
    fi

    "${PYTHON}" latent_eval/analyse_latents.py "${ANALYSIS_ARGS[@]}"
    echo ""
    echo "Analysis done. Results in: ${OUTPUT_DIR}/analysis"
fi
