#!/usr/bin/env bash
#
# Snellius job: download/generate ogbench/cube_double_expert.h5 dataset.
#
# Unlike cube_single (available on HuggingFace), the cube_double pixel
# dataset must be generated locally by rolling out the built-in expert
# policy. Uses stable_worldmodel's World.record_dataset() to collect
# pixel-based demonstrations for the OGBench Cube Double environment
# (2 cubes to manipulate) and saves them to:
#
#   $STABLEWM_HOME/ogbench/cube_double_expert.h5
#
# The dataset format matches what stable_worldmodel's HDF5Dataset expects,
# so it can be used directly for LeWM training with data=hi_ogb_double.
#
# Usage:
#   cd jobs/setup
#   sbatch download_cube_double.sh
#
# Optional overrides:
#   sbatch --export=ALL,N_EPISODES=2000 download_cube_double.sh
#   sbatch --export=ALL,N_ENVS=16 download_cube_double.sh
#   sbatch --export=ALL,STABLEWM_HOME=/scratch-shared/$USER/stablewm_data download_cube_double.sh

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=download_cube_double
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=download_cube_double_%j.out
#SBATCH --error=download_cube_double_%j.err

set -euo pipefail

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
  echo "ERROR: conda environment 'lewm-gpu' or 'lewm' not found." >&2
  exit 2
fi
set -u

export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"
export N_EPISODES="${N_EPISODES:-10000}"
export N_ENVS="${N_ENVS:-8}"
export DATASET_NAME="ogbench/cube_double_expert"
export SEED="${SEED:-42}"
export MUJOCO_GL=egl

echo "STABLEWM_HOME: ${STABLEWM_HOME}"
echo "Dataset name:  ${DATASET_NAME}"
echo "Output file:   ${STABLEWM_HOME}/${DATASET_NAME}.h5"
echo "Episodes:      ${N_EPISODES}"
echo "Parallel envs: ${N_ENVS}"
echo "Seed:          ${SEED}"
echo ""

mkdir -p "${STABLEWM_HOME}/ogbench"

python - <<PYEOF
import os, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench.expert_policy import ExpertPolicy

STABLEWM_HOME = os.environ['STABLEWM_HOME']
N_EPISODES    = int(os.environ['N_EPISODES'])
N_ENVS        = int(os.environ['N_ENVS'])
DATASET_NAME  = os.environ['DATASET_NAME']
SEED          = int(os.environ['SEED'])

print(f"Creating Cube Double world with {N_ENVS} parallel environments...")
world = swm.World(
    'swm/OGBCube-v0',
    num_envs=N_ENVS,
    image_shape=(224, 224),
    max_episode_steps=200,
    env_type='double',
    mode='data_collection',
)

print("Setting up expert policy...")
expert = ExpertPolicy(policy_type='markov_oracle', action_noise=0.1, seed=SEED)
expert.set_env(world.envs)
world.set_policy(expert)

print(f"Recording {N_EPISODES} episodes into {DATASET_NAME}...")
world.record_dataset(
    dataset_name=DATASET_NAME,
    episodes=N_EPISODES,
    seed=SEED,
    cache_dir=STABLEWM_HOME,
)

output_path = os.path.join(STABLEWM_HOME, f'{DATASET_NAME}.h5')
import h5py
with h5py.File(output_path, 'r') as f:
    print(f"Dataset saved: {output_path}")
    print(f"Keys: {list(f.keys())}")
    for k in f.keys():
        print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")
PYEOF

echo ""
echo "Done. Dataset at: ${STABLEWM_HOME}/${DATASET_NAME}.h5"
