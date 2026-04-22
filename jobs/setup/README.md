# Setup Jobs

Environment and dataset setup jobs for cluster runs.

## Scripts

- `setup_env.sh`: create/update the conda environment from `environment-gpu.yml`.
- `test_env.sh`: verify environment + CUDA/PyTorch imports.
- `download_pusht.sh`: download PushT (or selected datasets via `DATASETS=...`).
- `download_cube.sh`: download Cube dataset.
- `download_reacher.sh`: download Reacher dataset.
- `download_tworooms.sh`: download TwoRooms dataset.
