# Experiments

The canonical experiment seam is:

```bash
python -m h_le_wm.experiments.run --spec <name>
```

Thin shell wrappers in `scripts/` are the reader-facing shortcuts.

## Public specs

| Spec | Wrapper | Required named checkpoints | Output root |
| --- | --- | --- | --- |
| `matrix/pusht/baseline` | `scripts/run_pusht_baseline_matrix.sh` | `baseline/pusht/lewm` | `STABLEWM_HOME/repro/matrix__pusht__baseline` |
| `matrix/pusht/hierarchical` | `scripts/run_pusht_hierarchical_matrix.sh` | `hierarchical/pusht/hope2_epoch15` | `STABLEWM_HOME/repro/matrix__pusht__hierarchical` |
| `matrix/cube/baseline` | `scripts/run_cube_baseline_matrix.sh` | `baseline/cube/lewm` | `STABLEWM_HOME/repro/matrix__cube__baseline` |
| `matrix/cube/hierarchical` | `scripts/run_cube_hierarchical_matrix.sh` | `hierarchical/cube/hope2_epoch15` | `STABLEWM_HOME/repro/matrix__cube__hierarchical` |
| `smoke/pusht` | `scripts/run_pusht_smoke.sh` | `baseline/pusht/lewm` | `STABLEWM_HOME/runs/pusht_smoke` and `STABLEWM_HOME/repro/pusht_smoke` |
| `diagnostics/pusht/offline` | `scripts/run_pusht_offline_diagnostics.sh` | `hierarchical/pusht/hope2_epoch15` | `STABLEWM_HOME/repro/pusht_offline_diagnostics` |
| `diagnostics/pusht/acting` | `scripts/run_pusht_acting_diagnostics.sh` | `hierarchical/pusht/hope2_epoch15` | `STABLEWM_HOME/repro/pusht_acting_diagnostics` |
| `probe/pusht/phase_a/train` | `scripts/train_pusht_probe_phase_a.sh` | `hierarchical/pusht/hope2_epoch15` | `STABLEWM_HOME/runs/pusht_probe_phase_a` |
| `probe/pusht/phase_b/train` | `scripts/train_pusht_probe_phase_b.sh` | `hierarchical/pusht/hope2_epoch15` and Phase A output | `STABLEWM_HOME/runs/pusht_probe_phase_b` |
| `render/pusht/paper_diagnostics` | `scripts/render_pusht_paper_diagnostics.sh` | `hierarchical/pusht/hope2_epoch15` | `STABLEWM_HOME/repro/pusht_paper_diagnostics_render` |
| `render/pusht/decoder_story_figures` | `scripts/render_pusht_decoder_story_figures.sh` | `hierarchical/pusht/hope2_epoch15` and `probe/pusht/phase_b` | `STABLEWM_HOME/repro/pusht_decoder_story_figures` |
| `render/pusht/story_figures` | `scripts/render_pusht_story_figures.sh` | `hierarchical/pusht/hope2_epoch15` and `probe/pusht/phase_b` | `STABLEWM_HOME/repro/pusht_story_figures` |
| `paper/reproduction` | `scripts/run_paper_reproduction.sh` | required-now and supported-first-class surface | `STABLEWM_HOME/repro/paper_reproduction` |

## Notes

- Public renders and workflows depend on named upstream outputs, not hand-passed cluster-era files.
- The probe training specs are first-class, but the default paper reproduction workflow consumes staged probe bundles rather than retraining them.
- `jobs/` and `roadmap/` remain internal reference material and are not part of the supported experiment interface.
