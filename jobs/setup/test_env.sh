#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=TestEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00

set -eo pipefail

# Setup logging
module purge
module load 2025
module load Anaconda3/2025.06-1

resolve_repo_root() {
  local c p
  for c in \
    "${PROJECT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${HOME}/h-lewm" \
    "${HOME}/h-le-wm" \
    "/gpfs/home2/${USER}/h-lewm" \
    "/gpfs/home2/${USER}/h-le-wm"; do
    [[ -z "${c}" ]] && continue
    for p in "${c}" "${c}/.." "${c}/../.."; do
      if p="$(cd "${p}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${p}/environment-gpu.yml" ]]; then
          echo "${p}"
          return 0
        fi
      fi
    done
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate repo root with environment-gpu.yml" >&2
  exit 2
fi

LOG_DIR="${REPO_ROOT}/jobs/setup/out"
mkdir -p "${LOG_DIR}"
JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%s)}"
exec > >(tee -a "${LOG_DIR}/test_env_${JOB_TAG}.out") \
     2> >(tee -a "${LOG_DIR}/test_env_${JOB_TAG}.err" >&2)

echo "========================================"
echo "Environment Test Script"
echo "========================================"
echo "Repo root: ${REPO_ROOT}"
echo "Job tag: ${JOB_TAG}"
echo "Test time: $(date)"
echo ""

# Test 1: Check if environment exists
echo "[TEST 1] Checking if 'lewm-gpu' environment exists..."
if conda env list | grep -q "lewm-gpu"; then
  echo "✓ Environment 'lewm-gpu' found"
else
  echo "✗ FAILED: Environment 'lewm-gpu' not found"
  echo "Available environments:"
  conda env list
  exit 1
fi
echo ""

# Test 2: Activate environment and check Python
echo "[TEST 2] Testing Python version..."
eval "$(conda shell.bash hook)"
conda activate lewm-gpu
PYTHON_VERSION=$(python --version 2>&1)
echo "Python version: $PYTHON_VERSION"
if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
  echo "✓ Python version >= 3.10"
else
  echo "✗ FAILED: Python version is not >= 3.10"
  exit 1
fi
echo ""

# Test 3: Check core packages
echo "[TEST 3] Checking core packages..."
PACKAGES=("numpy" "torch" "cuda")
for pkg in "${PACKAGES[@]}"; do
  if python -c "import ${pkg}" 2>/dev/null; then
    echo "✓ Package '${pkg}' is installed"
  else
    echo "✗ WARNING: Package '${pkg}' not found"
  fi
done
echo ""

# Test 4: Check CUDA availability
echo "[TEST 4] Testing CUDA availability..."
if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null; then
  echo "✓ CUDA check completed"
else
  echo "✗ WARNING: Could not verify CUDA (torch may not be installed)"
fi
echo ""

# Test 5: Check stable-worldmodel package
echo "[TEST 5] Checking 'stable-worldmodel' package..."
if python -c "import stable_worldmodel" 2>/dev/null; then
  echo "✓ Package 'stable-worldmodel' is installed"
  python -c "import stable_worldmodel; print(f'  Version info available')" 2>/dev/null || true
else
  echo "✗ FAILED: Package 'stable-worldmodel' is not installed"
  exit 1
fi
echo ""

# Test 6: Verify pip and essential tools
echo "[TEST 6] Checking pip and build tools..."
if pip --version > /dev/null 2>&1; then
  echo "✓ pip is available: $(pip --version)"
else
  echo "✗ FAILED: pip is not available"
  exit 1
fi
echo ""

# Test 7: Import test for project modules
echo "[TEST 7] Testing project module imports..."
cd "${REPO_ROOT}"
if python -c "from hi_module import *" 2>/dev/null; then
  echo "✓ Project modules can be imported"
else
  echo "✗ WARNING: Could not import project modules (this may be expected)"
fi
echo ""

echo "========================================"
echo "✓ All critical tests passed!"
echo "========================================"
echo "Environment setup is complete and functional."
