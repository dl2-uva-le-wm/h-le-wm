# Checkpoints

Checkpoint setup is explicit local staging only. The repo does not download or register checkpoints for you.

## Required-now checkpoints

These are required by the default paper preflight:

- `baseline/pusht/lewm` -> `STABLEWM_HOME/pusht/lewm_object.ckpt`
- `baseline/cube/lewm` -> `STABLEWM_HOME/cube/lewm_object.ckpt`

Stage them with:

```bash
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint baseline/pusht/lewm=/absolute/path/to/pusht_lewm_object.ckpt \
  --checkpoint baseline/cube/lewm=/absolute/path/to/cube_lewm_object.ckpt
```

Use `--dry-run` to verify placements and `--force` to replace an existing staged file.

## Supported-first-class checkpoints

The registry also tracks stable names for:

- `hierarchical/pusht/hope2_epoch15` -> `STABLEWM_HOME/runs/pusht_hierarchical_hope2/pusht_hierarchical_hope2_epoch_15_object.ckpt`
- `hierarchical/cube/hope2_epoch15` -> `STABLEWM_HOME/runs/cube_hierarchical_hope2/cube_hierarchical_hope2_epoch_15_object.ckpt`
- `probe/pusht/phase_a` -> `STABLEWM_HOME/runs/pusht_probe_phase_a/pusht_probe_phase_a_probe.pt`
- `probe/pusht/phase_b` -> `STABLEWM_HOME/runs/pusht_probe_phase_b/pusht_probe_phase_b_probe.pt`

Stage the full reader-facing surface with:

```bash
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint baseline/pusht/lewm=/absolute/path/to/pusht_lewm_object.ckpt \
  --checkpoint baseline/cube/lewm=/absolute/path/to/cube_lewm_object.ckpt \
  --checkpoint hierarchical/pusht/hope2_epoch15=/absolute/path/to/pusht_hope2_epoch15_object.ckpt \
  --checkpoint hierarchical/cube/hope2_epoch15=/absolute/path/to/cube_hope2_epoch15_object.ckpt \
  --checkpoint probe/pusht/phase_a=/absolute/path/to/pusht_probe_phase_a_probe.pt \
  --checkpoint probe/pusht/phase_b=/absolute/path/to/pusht_probe_phase_b_probe.pt
```

The older `scripts/setup_checkpoints.sh` wrapper remains as a compatibility alias.

Inspect the machine-readable registry with:

```bash
python -m h_le_wm.checkpoints list
python -m h_le_wm.checkpoints list --tier supported-first-class
```

Validate the required-now checkpoint tier:

```bash
python -m h_le_wm.validate checkpoints
```

Validate the broader named surface:

```bash
python -m h_le_wm.validate checkpoints --tier supported-first-class
```
