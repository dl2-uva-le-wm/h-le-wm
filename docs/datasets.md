# Datasets

The milestone-two public setup path stages only the paper datasets:

- `pusht`
- `cube`

Use the canonical wrapper so `STABLEWM_HOME` is exported in your current shell:

```bash
source scripts/setup_paper_datasets.sh
```

To override the storage root:

```bash
source scripts/setup_paper_datasets.sh --home /absolute/path/to/stablewm
```

Validate dataset presence with:

```bash
python -m h_le_wm.validate datasets
```

This wrapper intentionally does not advertise `TwoRooms` or `Reacher` as part of the public paper path.
