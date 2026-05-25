# Architecture

## Boundary

- `third_party/lewm/` is the frozen upstream baseline boundary
- `h_le_wm/` is the canonical repo-owned namespace
- historical launchers and notes are not part of the public surface

## Package layout

- `h_le_wm/baseline/`
  - baseline delegation and vendored-module loading helpers
- `h_le_wm/models/`
  - hierarchical model components and waypoint sampling
- `h_le_wm/planning/`
  - hierarchical planning policies and empirical macro-action search
- `h_le_wm/train/`
  - canonical hierarchical training entrypoint
- `h_le_wm/eval/`
  - canonical hierarchical eval entrypoint and baseline manifest eval helper
- `h_le_wm/probe/`
  - decoder-probe model, training, and eval entrypoints
- `h_le_wm/experiments/`
  - experiment registry, specs, matrix inputs, and generic runner for smoke, diagnostics, renders, and paper workflows

## Command model

The documented interface is:

- `python -m h_le_wm.validate ...`
- `python -m h_le_wm.experiments.run ...`
- `python -m h_le_wm.train.hierarchical`
- `python -m h_le_wm.eval.hierarchical`
- `python -m h_le_wm.probe.train`
- `python -m h_le_wm.probe.eval`

Small wrapper scripts under `scripts/` exist for paper-facing setup, matrices, smoke, canonical training, diagnostics, probe training, render flows, and paper workflows.

## Public wrappers

- Dataset setup: `scripts/setup_paper_datasets.sh`
- Baseline checkpoint setup: `scripts/setup_baseline_checkpoints.sh`
- Smoke: `scripts/run_pusht_smoke.sh`
- Canonical training: `scripts/train_pusht_hierarchical_default.sh`, `scripts/train_cube_hierarchical_default.sh`
- Matrix evaluation: `scripts/run_pusht_baseline_matrix.sh`, `scripts/run_pusht_hierarchical_matrix.sh`, `scripts/run_cube_baseline_matrix.sh`, `scripts/run_cube_hierarchical_matrix.sh`
- Diagnostics and renders: `scripts/run_pusht_offline_diagnostics.sh`, `scripts/run_pusht_acting_diagnostics.sh`, `scripts/render_pusht_paper_diagnostics.sh`, `scripts/render_pusht_decoder_story_figures.sh`, `scripts/render_pusht_story_figures.sh`
- Probe training: `scripts/train_pusht_probe_phase_a.sh`, `scripts/train_pusht_probe_phase_b.sh`
- Paper workflows: `scripts/run_paper_reproduction.sh`, `scripts/run_paper_from_scratch.sh`

## Experiment system

- The public seam is `h_le_wm/experiments/index.yaml` plus `python -m h_le_wm.experiments.run --spec ...`
- Public specs are curated names such as `smoke/pusht`, `train/pusht/hierarchical_default`, `diagnostics/pusht/offline`, `render/pusht/story_figures`, `paper/reproduction`, and `paper/from_scratch`
- Command specs shell out to existing package modules or heavyweight render scripts
- Workflow specs compose first-class flows without exposing raw ad hoc paths

## Checkpoint and output contract

- Named checkpoints are staged under one `STABLEWM_HOME`
- Checkpoint-producing specs write under `STABLEWM_HOME/runs/...`
- Derived artifacts write under `STABLEWM_HOME/repro/...`
- The matrix surfaces keep their machine-readable summaries under deterministic `repro` roots
- Render specs consume named upstream outputs, not hand-passed cluster-era paths

## Paper reproduction flow

- Install the conda environment and stage `pusht` and `cube`
- Stage required-now baseline checkpoints for preflight and smoke
- Stage supported-first-class hierarchical and probe bundles for reader-facing reproduction
- Run `scripts/validate_preflight.sh`
- Run `scripts/run_pusht_smoke.sh` for the small GPU smoke path
- Run `scripts/run_paper_reproduction.sh` for checkpoint-driven paper reproduction
- Run `scripts/run_paper_from_scratch.sh` to retrain the canonical hierarchical and probe bundles before rerunning the paper graph

## Root layout

Root-level Python implementation shims are intentionally removed in the cleaned-up package-first layout.

- repo-owned implementation lives under `h_le_wm/`
- shell wrappers under `scripts/` remain as thin entrypoints into the package-first surface
- repo root should mostly contain metadata, environments, docs, and top-level folders
