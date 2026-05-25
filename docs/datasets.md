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

To stage a different supported subset through the lower-level wrapper:

```bash
source scripts/setup_datasets.sh --datasets pusht
source scripts/setup_datasets.sh --datasets cube
```

## Wrapper options

- `scripts/setup_paper_datasets.sh`
  Always stages `pusht,cube` and accepts `--home PATH`.
- `scripts/setup_datasets.sh`
  Accepts `--home PATH` and `--datasets pusht,cube,tworooms,reacher,all`.
- `python -m h_le_wm.validate datasets --datasets pusht,cube`
  Checks that the named dataset files exist under `STABLEWM_HOME`.

Validate dataset presence with:

```bash
python -m h_le_wm.validate datasets
```

This wrapper intentionally does not advertise `TwoRooms` or `Reacher` as part of the public paper path.
