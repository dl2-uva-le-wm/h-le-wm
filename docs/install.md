# Install

The canonical setup path is conda-based.

## Baseline environment

```bash
conda env create -f environment.yml
conda activate lewm
```

## GPU environment

```bash
conda env create -f environment-gpu.yml
conda activate lewm-gpu
```

## Supported contract

- Formal reproducibility target: Linux `x86_64` with CUDA
- `environment.yml`: developer and Tier 0 baseline
- `environment-gpu.yml`: Tier 1 and Tier 2 starting point
- macOS and ad hoc local setups are best-effort developer environments

Validate the installed surface with:

```bash
python -m h_le_wm.validate env
```
