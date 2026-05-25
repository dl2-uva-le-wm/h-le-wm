# Reproduction

This is the normative reader-facing path from a fresh clone to the supported paper artifact.

## 1. Install

```bash
git submodule update --init --recursive
conda env create -f environment-gpu.yml
conda activate lewm-gpu
```

If you are only doing Tier 0 setup checks, `environment.yml` is sufficient. The smoke path and the full reproduction path target the GPU environment.

## 2. Stage datasets

```bash
source scripts/setup_paper_datasets.sh
```

The public paper path stages only:

- `pusht`
- `cube`

## 3. Stage checkpoints

Required-now baseline checkpoints for setup and smoke:

```bash
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint baseline/pusht/lewm=/absolute/path/to/pusht_lewm_object.ckpt \
  --checkpoint baseline/cube/lewm=/absolute/path/to/cube_lewm_object.ckpt
```

Supported-first-class checkpoints for full paper reproduction:

```bash
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint hierarchical/pusht/hope2_epoch15=/absolute/path/to/pusht_hope2_epoch15_object.ckpt \
  --checkpoint hierarchical/cube/hope2_epoch15=/absolute/path/to/cube_hope2_epoch15_object.ckpt \
  --checkpoint probe/pusht/phase_a=/absolute/path/to/pusht_probe_phase_a_probe.pt \
  --checkpoint probe/pusht/phase_b=/absolute/path/to/pusht_probe_phase_b_probe.pt
```

## 4. Validate

Run the non-expensive setup checks:

```bash
bash scripts/validate_preflight.sh
python -m h_le_wm.validate checkpoints --tier supported-first-class
```

## 5. Run the PushT smoke path

The smoke surface is a small GPU train+eval flow backed by the canonical spec graph:

```bash
bash scripts/run_pusht_smoke.sh
```

Use `--dry-run` first if you want to inspect the generated commands:

```bash
bash scripts/run_pusht_smoke.sh --dry-run
```

## 6. Run the full paper reproduction

The full reader-facing workflow is:

```bash
bash scripts/run_paper_reproduction.sh
```

This workflow covers:

- PushT baseline matrix
- PushT hierarchical matrix
- Cube baseline matrix
- Cube hierarchical matrix
- PushT offline diagnostics
- PushT acting diagnostics
- PushT paper diagnostics render
- PushT decoder story figures
- PushT story figures

All public first-class surfaces are spec-driven through `python -m h_le_wm.experiments.run --spec ...`.
