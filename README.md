# h-le-wm

Paper-ready H-LeWM workspace with a frozen upstream baseline in `third_party/lewm` and a canonical repo-owned namespace in `h_le_wm/`.

## Quickstart

Clone the repo and initialize the pinned upstream baseline:

```bash
git clone https://github.com/NiccoloCase/h-le-wm.git
cd h-le-wm
git submodule update --init --recursive
```

Create the canonical environment:

```bash
conda env create -f environment.yml
conda activate lewm
```

Or use the GPU-focused environment:

```bash
conda env create -f environment-gpu.yml
conda activate lewm-gpu
```

Stage the paper datasets and the required-now baseline checkpoints under `STABLEWM_HOME`:

```bash
source scripts/setup_paper_datasets.sh
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint baseline/pusht/lewm=/absolute/path/to/pusht_lewm_object.ckpt \
  --checkpoint baseline/cube/lewm=/absolute/path/to/cube_lewm_object.ckpt
```

Run the non-expensive paper preflight:

```bash
bash scripts/validate_preflight.sh
```

## Canonical Commands

The documented interface is package-first:

```bash
python -m h_le_wm.validate preflight
python -m h_le_wm.experiments.run --spec smoke/pusht --dry-run
python -m h_le_wm.experiments.run --spec paper/reproduction --dry-run
python -m h_le_wm.train.hierarchical --help
python -m h_le_wm.eval.hierarchical --help
python -m h_le_wm.probe.train --help
python -m h_le_wm.probe.eval --help
```

The supported Python entrypoints are the package modules under `h_le_wm/`.

## Reader Paths

Install + preflight:

```bash
source scripts/setup_paper_datasets.sh
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint baseline/pusht/lewm=/absolute/path/to/pusht_lewm_object.ckpt \
  --checkpoint baseline/cube/lewm=/absolute/path/to/cube_lewm_object.ckpt
bash scripts/validate_preflight.sh
```

PushT smoke:

```bash
bash scripts/run_pusht_smoke.sh --dry-run
```

Full paper reproduction:

```bash
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint baseline/pusht/lewm=/absolute/path/to/pusht_lewm_object.ckpt \
  --checkpoint baseline/cube/lewm=/absolute/path/to/cube_lewm_object.ckpt \
  --checkpoint hierarchical/pusht/hope2_epoch15=/absolute/path/to/pusht_hope2_epoch15_object.ckpt \
  --checkpoint hierarchical/cube/hope2_epoch15=/absolute/path/to/cube_hope2_epoch15_object.ckpt \
  --checkpoint probe/pusht/phase_a=/absolute/path/to/pusht_probe_phase_a_probe.pt \
  --checkpoint probe/pusht/phase_b=/absolute/path/to/pusht_probe_phase_b_probe.pt
python -m h_le_wm.validate checkpoints --tier supported-first-class
bash scripts/run_paper_reproduction.sh --dry-run
```

## Public Surface

Reader-facing setup covers:

- datasets: `pusht`, `cube`
- required-now checkpoints:
  - `baseline/pusht/lewm` for smoke and preflight
  - `baseline/cube/lewm` for preflight
- supported-first-class checkpoint registry:
  - the required-now baseline checkpoints above
  - canonical hierarchical PushT and Cube HOPE2 checkpoints
  - canonical PushT decoder-probe Phase A and Phase B bundles
- canonical specs:
  - matrix: `matrix/pusht/*`, `matrix/cube/*`
  - smoke: `smoke/pusht`
  - diagnostics: `diagnostics/pusht/offline`, `diagnostics/pusht/acting`
  - probe training: `probe/pusht/phase_a/train`, `probe/pusht/phase_b/train`
  - renders: `render/pusht/*`
  - workflow: `paper/reproduction`

List the named checkpoint slots at any time with:

```bash
python -m h_le_wm.checkpoints list
```

## Docs

- [Install](docs/install.md)
- [Datasets](docs/datasets.md)
- [Reproduction](docs/reproduction.md)
- [Hardware](docs/hardware.md)
- [Experiments](docs/experiments.md)
- [Checkpoints](docs/checkpoints.md)
- [Outputs](docs/outputs.md)
- [Architecture](docs/architecture.md)
- [Architecture Variants](docs/architecture_variants.md)
