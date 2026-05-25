# Checkpoints

Checkpoint setup supports either explicit local staging or direct baseline fetches from the official Hugging Face LeWM model pages.

## Required-now checkpoints

These are required by the default paper preflight:

- `baseline/pusht/lewm` -> `STABLEWM_HOME/pusht/lewm_object.ckpt`
- `baseline/cube/lewm` -> `STABLEWM_HOME/cube/lewm_object.ckpt`

Stage them with:

```bash
bash scripts/setup_baseline_checkpoints.sh fetch-baselines
```

Or with explicit local files:

```bash
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint baseline/pusht/lewm=/absolute/path/to/pusht_lewm_object.ckpt \
  --checkpoint baseline/cube/lewm=/absolute/path/to/cube_lewm_object.ckpt
```

Use `--dry-run` to verify placements and `--force` to replace an existing staged file.

The Hugging Face fetch path downloads the official `config.json` + `weights.pt` model artifacts and converts them into the canonical `*_object.ckpt` files expected by this repo.

## Supported-first-class checkpoints

The registry also tracks stable names for:

- `hierarchical/pusht/default_epoch15` -> `STABLEWM_HOME/runs/pusht_hierarchical_default/pusht_hierarchical_default_epoch_15_object.ckpt`
- `hierarchical/cube/default_epoch15` -> `STABLEWM_HOME/runs/cube_hierarchical_default/cube_hierarchical_default_epoch_15_object.ckpt`
- `probe/pusht/phase_a` -> `STABLEWM_HOME/runs/pusht_probe_phase_a/pusht_probe_phase_a_probe.pt`
- `probe/pusht/phase_b` -> `STABLEWM_HOME/runs/pusht_probe_phase_b/pusht_probe_phase_b_probe.pt`

Stage the full reader-facing surface with:

```bash
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint baseline/pusht/lewm=/absolute/path/to/pusht_lewm_object.ckpt \
  --checkpoint baseline/cube/lewm=/absolute/path/to/cube_lewm_object.ckpt \
  --checkpoint hierarchical/pusht/default_epoch15=/absolute/path/to/pusht_default_epoch15_object.ckpt \
  --checkpoint hierarchical/cube/default_epoch15=/absolute/path/to/cube_default_epoch15_object.ckpt \
  --checkpoint probe/pusht/phase_a=/absolute/path/to/pusht_probe_phase_a_probe.pt \
  --checkpoint probe/pusht/phase_b=/absolute/path/to/pusht_probe_phase_b_probe.pt
```

## Command options

- `bash scripts/setup_baseline_checkpoints.sh fetch-baselines`
  Downloads and converts the official baseline models for the required-now tier.
- `bash scripts/setup_baseline_checkpoints.sh stage --checkpoint name=/path`
  Stages explicit local files into canonical registry locations.
- `python -m h_le_wm.checkpoints list [--tier TIER]`
  Prints the machine-readable checkpoint registry.
- `--dry-run`
  Prints the target paths without copying or downloading.
- `--force`
  Replaces an already-staged target path.

The older `scripts/setup_checkpoints.sh` wrapper remains as a compatibility alias.

Inspect the machine-readable registry with:

```bash
python -m h_le_wm.checkpoints list
python -m h_le_wm.checkpoints list --tier supported-first-class
```

Preview the Hugging Face fetch plan without downloading:

```bash
python -m h_le_wm.checkpoints fetch-baselines --dry-run
```

Validate the required-now checkpoint tier:

```bash
python -m h_le_wm.validate checkpoints
```

Validate the broader named surface:

```bash
python -m h_le_wm.validate checkpoints --tier supported-first-class
```
