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

## Command choices

- Use `environment.yml` for Tier 0 validation and doc inspection.
- Use `environment-gpu.yml` for the smoke path, matrix evaluation, diagnostics, renders, and full reproduction.
- Use `bash scripts/run_pusht_smoke.sh --dry-run` to inspect the smallest GPU-backed workflow before running it.

## Full paper path

The first-class matrix, diagnostics, and render surfaces assume the supported CUDA platform above. The repo documents one canonical environment and one canonical checkpoint/output contract rather than multiple hardware variants.
