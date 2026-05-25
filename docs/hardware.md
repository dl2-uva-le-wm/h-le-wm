# Hardware

## Supported platform

The formal reproducibility target is:

- Linux `x86_64`
- CUDA-enabled NVIDIA GPU
- the pinned conda environment from `environment-gpu.yml`

macOS and ad hoc local setups remain best-effort developer environments, not part of the formal guarantee.

## Smoke path

`scripts/run_pusht_smoke.sh` is the smallest supported GPU path.

- one GPU
- one training epoch
- small eval budget
- small `num_eval`
- `wandb` disabled by default

## Full paper path

The first-class matrix, diagnostics, and render surfaces assume the supported CUDA platform above. The repo documents one canonical environment and one canonical checkpoint/output contract rather than multiple hardware variants.
