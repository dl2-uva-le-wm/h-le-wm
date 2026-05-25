# Reproduction

This is the authoritative reader-facing playbook for the two supported paths:

- reproduce the paper artifact from staged canonical checkpoints
- produce fresh paper-shaped results by retraining the canonical checkpoints, then rerunning the paper workflow

The formal target is Linux `x86_64` with a CUDA-capable NVIDIA GPU and `environment-gpu.yml`.

## Common Setup

### 1. Clone and pin the baseline

```bash
git clone https://github.com/NiccoloCase/h-le-wm.git
cd h-le-wm
git submodule update --init --recursive
```

Verify:

```bash
python -m h_le_wm.validate baseline
```

### 2. Create the GPU environment

```bash
conda env create -f environment-gpu.yml
conda activate lewm-gpu
```

Verify:

```bash
python -m h_le_wm.validate env
```

### 3. Stage the paper datasets

```bash
source scripts/setup_paper_datasets.sh
```

This stages only the public paper datasets:

- `pusht`
- `cube`

Verify:

```bash
python -m h_le_wm.validate datasets --datasets pusht,cube
```

## Artifact Reproduction

Use this path when you already have the canonical hierarchical and probe checkpoints and want to reproduce the paper artifact quickly.

### 1. Stage the required-now baseline checkpoints

Fetch the official baseline checkpoints:

```bash
bash scripts/setup_baseline_checkpoints.sh fetch-baselines
```

Or stage them manually:

```bash
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint baseline/pusht/lewm=/absolute/path/to/pusht_lewm_object.ckpt \
  --checkpoint baseline/cube/lewm=/absolute/path/to/cube_lewm_object.ckpt
```

Verify:

```bash
python -m h_le_wm.validate checkpoints --tier required-now
```

### 2. Stage the supported-first-class hierarchical and probe bundles

```bash
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint hierarchical/pusht/default_epoch15=/absolute/path/to/pusht_default_epoch15_object.ckpt \
  --checkpoint hierarchical/cube/default_epoch15=/absolute/path/to/cube_default_epoch15_object.ckpt \
  --checkpoint probe/pusht/phase_a=/absolute/path/to/pusht_probe_phase_a_probe.pt \
  --checkpoint probe/pusht/phase_b=/absolute/path/to/pusht_probe_phase_b_probe.pt
```

Verify:

```bash
python -m h_le_wm.validate checkpoints --tier supported-first-class
```

### 3. Run the non-expensive preflight

```bash
bash scripts/validate_preflight.sh
```

This verifies the environment, pinned baseline, public datasets, and required-now baseline checkpoints.

### 4. Optional smoke path

Preview the commands first:

```bash
bash scripts/run_pusht_smoke.sh --dry-run
```

Run the smoke workflow:

```bash
bash scripts/run_pusht_smoke.sh
```

Verify:

```bash
python -m h_le_wm.validate outputs --spec smoke/pusht
```

### 5. Run the full paper workflow

```bash
bash scripts/run_paper_reproduction.sh
```

Verify:

```bash
python -m h_le_wm.validate outputs --spec paper/reproduction
```

### 6. Inspect the stable roots

The canonical roots to inspect after artifact reproduction are:

- `STABLEWM_HOME/runs/pusht_hierarchical_default`
- `STABLEWM_HOME/runs/cube_hierarchical_default`
- `STABLEWM_HOME/runs/pusht_probe_phase_a`
- `STABLEWM_HOME/runs/pusht_probe_phase_b`
- `STABLEWM_HOME/repro/paper_reproduction`

## Fresh Results

Use this path when you want the repo to train the canonical hierarchical and probe checkpoints from the baseline models, then rerun the paper reproduction workflow against those new artifacts.

### 1. Stage the baseline checkpoints

```bash
bash scripts/setup_baseline_checkpoints.sh fetch-baselines
```

Verify:

```bash
python -m h_le_wm.validate checkpoints --tier required-now
```

### 2. Run the non-expensive preflight

```bash
bash scripts/validate_preflight.sh
```

### 3. Optional smoke path

Preview:

```bash
bash scripts/run_pusht_smoke.sh --dry-run
```

Run:

```bash
bash scripts/run_pusht_smoke.sh
```

Verify:

```bash
python -m h_le_wm.validate outputs --spec smoke/pusht
```

### 4. Run the canonical from-scratch workflow

```bash
bash scripts/run_paper_from_scratch.sh
```

This workflow runs, in order:

- `train/pusht/hierarchical_default`
- `train/cube/hierarchical_default`
- `probe/pusht/phase_a/train`
- `probe/pusht/phase_b/train`
- `paper/reproduction`

Verify:

```bash
python -m h_le_wm.validate outputs --spec paper/from_scratch
```

### 5. Inspect the stable roots

The canonical roots to inspect after the from-scratch path are:

- `STABLEWM_HOME/runs/pusht_hierarchical_default`
- `STABLEWM_HOME/runs/cube_hierarchical_default`
- `STABLEWM_HOME/runs/pusht_probe_phase_a`
- `STABLEWM_HOME/runs/pusht_probe_phase_b`
- `STABLEWM_HOME/repro/paper_reproduction`

## Notes

- `STABLEWM_HOME` defaults to `data/stablewm` when not set explicitly.
- `environment.yml` remains useful for Tier 0 validation and documentation work, but the supported reproduction target is `environment-gpu.yml`.
- The public workflow surface is spec-driven through `python -m h_le_wm.experiments.run --spec ...`.
