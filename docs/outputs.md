# Outputs

The canonical output contract is split in two:

- `STABLEWM_HOME/runs/...` for checkpoint-producing specs
- `STABLEWM_HOME/repro/...` for derived evaluation, diagnostics, summaries, and renders

## Canonical run roots

- `STABLEWM_HOME/runs/pusht_smoke`
- `STABLEWM_HOME/runs/pusht_probe_phase_a`
- `STABLEWM_HOME/runs/pusht_probe_phase_b`
- `STABLEWM_HOME/runs/pusht_hierarchical_default`
- `STABLEWM_HOME/runs/cube_hierarchical_default`

## Canonical repro roots

- `STABLEWM_HOME/repro/matrix__pusht__baseline`
- `STABLEWM_HOME/repro/matrix__pusht__hierarchical`
- `STABLEWM_HOME/repro/matrix__cube__baseline`
- `STABLEWM_HOME/repro/matrix__cube__hierarchical`
- `STABLEWM_HOME/repro/pusht_smoke`
- `STABLEWM_HOME/repro/pusht_offline_diagnostics`
- `STABLEWM_HOME/repro/pusht_acting_diagnostics`
- `STABLEWM_HOME/repro/pusht_paper_diagnostics_render`
- `STABLEWM_HOME/repro/pusht_decoder_story_figures`
- `STABLEWM_HOME/repro/pusht_story_figures`
- `STABLEWM_HOME/repro/paper_reproduction`

## Machine-readable artifacts

- Matrix specs write `raw_rows.csv`, `summary.csv`, and a completion file under their deterministic repro root
- Diagnostics write per-run `.json` and `.npz` artifacts plus summary `.tsv` files under their deterministic repro root
- Render specs write image and table artifacts plus a manifest under their deterministic repro root

## Navigation rules

- Prefer named specs, named checkpoints, and these stable roots over ad hoc files.
- Reader-facing workflows should treat `runs/` as checkpoint-producing roots and `repro/` as derived-artifact roots.
- Use `python -m h_le_wm.validate outputs --spec <name>` to verify completion markers and deterministic output artifacts for a public spec or workflow.
