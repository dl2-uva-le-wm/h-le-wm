# h-le-wm

Reproducibility-first H-LeWM workspace with a frozen upstream baseline in `third_party/lewm` and a canonical repo-owned surface in `h_le_wm/`.

## Supported Platform

The formal reproduction target is:

- Linux `x86_64`
- CUDA-capable NVIDIA GPU
- `environment-gpu.yml`

Use `environment.yml` only for Tier 0 setup checks and doc inspection.

## Reader Paths

### Verify Setup

```bash
git clone https://github.com/NiccoloCase/h-le-wm.git
cd h-le-wm
git submodule update --init --recursive
conda env create -f environment-gpu.yml
conda activate lewm-gpu
source scripts/setup_paper_datasets.sh
bash scripts/setup_baseline_checkpoints.sh fetch-baselines
bash scripts/validate_preflight.sh
```

### Reproduce The Paper From Staged Checkpoints

```bash
source scripts/setup_paper_datasets.sh
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint baseline/pusht/lewm=/absolute/path/to/pusht_lewm_object.ckpt \
  --checkpoint baseline/cube/lewm=/absolute/path/to/cube_lewm_object.ckpt \
  --checkpoint hierarchical/pusht/default_epoch15=/absolute/path/to/pusht_default_epoch15_object.ckpt \
  --checkpoint hierarchical/cube/default_epoch15=/absolute/path/to/cube_default_epoch15_object.ckpt \
  --checkpoint probe/pusht/phase_a=/absolute/path/to/pusht_probe_phase_a_probe.pt \
  --checkpoint probe/pusht/phase_b=/absolute/path/to/pusht_probe_phase_b_probe.pt
python -m h_le_wm.validate checkpoints --tier supported-first-class
bash scripts/run_paper_reproduction.sh
python -m h_le_wm.validate outputs --spec paper/reproduction
```

### Produce Fresh Results From Retraining

```bash
source scripts/setup_paper_datasets.sh
bash scripts/setup_baseline_checkpoints.sh fetch-baselines
bash scripts/validate_preflight.sh
bash scripts/run_paper_from_scratch.sh
python -m h_le_wm.validate outputs --spec paper/from_scratch
```

## Smoke Check

Preview the smallest supported GPU path:

```bash
bash scripts/run_pusht_smoke.sh --dry-run
```

Run it:

```bash
bash scripts/run_pusht_smoke.sh
python -m h_le_wm.validate outputs --spec smoke/pusht
```

## Stable Outputs

- `STABLEWM_HOME` defaults to `data/stablewm`
- canonical checkpoint-producing roots live under `STABLEWM_HOME/runs/...`
- canonical derived artifacts live under `STABLEWM_HOME/repro/...`
- the main reader-facing roots to inspect are:
  - `STABLEWM_HOME/runs/pusht_hierarchical_default`
  - `STABLEWM_HOME/runs/cube_hierarchical_default`
  - `STABLEWM_HOME/runs/pusht_probe_phase_a`
  - `STABLEWM_HOME/runs/pusht_probe_phase_b`
  - `STABLEWM_HOME/repro/paper_reproduction`

`docs/reproduction.md` is the authoritative end-to-end playbook.

## Docs

- [Install](docs/install.md)
- [Datasets](docs/datasets.md)
- [Train Hierarchical Models](docs/train_hierarchical.md)
- [Reproduction](docs/reproduction.md)
- [Hardware](docs/hardware.md)
- [Experiments](docs/experiments.md)
- [Checkpoints](docs/checkpoints.md)
- [Outputs](docs/outputs.md)
- [Architecture](docs/architecture.md)
- [Architecture Variants](docs/architecture_variants.md)
