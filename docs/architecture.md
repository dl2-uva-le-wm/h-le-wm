# Architecture

## Boundary

- `third_party/lewm/` is the frozen upstream baseline boundary
- `h_le_wm/` is the canonical repo-owned namespace
- `jobs/` and `roadmap/` remain source material and internal history, not the public surface

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

Small wrapper scripts under `scripts/` exist for paper-facing setup, matrices, smoke, diagnostics, probe training, render flows, and paper reproduction.

## Experiment system

- The public seam is `h_le_wm/experiments/index.yaml` plus `python -m h_le_wm.experiments.run --spec ...`
- Public specs are curated names such as `smoke/pusht`, `diagnostics/pusht/offline`, `render/pusht/story_figures`, and `paper/reproduction`
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
- Run `scripts/run_paper_reproduction.sh` for the full first-class matrix, diagnostics, and render graph

## Root layout

Root-level Python implementation shims are intentionally removed in the cleaned-up package-first layout.

- repo-owned implementation lives under `h_le_wm/`
- shell wrappers under `scripts/` remain as thin entrypoints into the package-first surface
- repo root should mostly contain metadata, environments, docs, and top-level folders
