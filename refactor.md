# Refactor Plan

## Goal

Turn this repository into a paper-ready, fully reproducible experiment repo with the following contract:

> From a fresh clone, a user can install the project, download the required datasets, run a small smoke experiment locally on a supported GPU, run the full paper experiments on documented hardware using pinned configs and seeds, and regenerate the paper tables and plots from rerun outputs.

The repo should be clean, modular, and easy to navigate.

## Hard Constraints

- `third_party/` stays untouched.
- `jobs/` and `roadmap/` are not part of the shipped public surface.
- `jobs/` and `roadmap/` may be used as verified source material for the refactor.
- Code should move toward one canonical repo-owned namespace: `h_le_wm/`.
- The refactor does not require shipping a distributable/installable Python package.
- The final documented interface should be `python -m h_le_wm...` plus small paper-facing wrapper scripts.
- The shipped repo should be organized around reproducibility, not around cluster history.

## Scope

### First-Class, Claim-Bearing Surfaces

- claim-bearing curated matrix families are first-class experiments
- `PushT`
  - hierarchical training
  - baseline flat matrix
  - hierarchical matrix
  - offline diagnostics
  - acting diagnostics
  - decoder probe Phase A and Phase B
  - paper diagnostics render
  - decoder-story figures
  - story figures
- `Cube`
  - hierarchical training
  - baseline flat matrix
  - hierarchical matrix

### Explicitly Out of Scope for the Shipped Public Surface

- `TwoRooms`
- `Reacher`
- cluster-specific Slurm launchers
- notebooks as part of the formal reproducibility contract
- full baseline retraining as a required paper path
- dormant or exploratory branches outside the supported mainline variants

## Reproducibility Contract

### Tiers

- `Tier 0`: setup and validation checks
- `Tier 1`: a small GPU smoke run
- `Tier 2`: full paper reproduction on documented hardware

### Checkpoints

- Reader-facing paper reproduction is `checkpoints-only`.
- Users are expected to place required checkpoints under one `STABLEWM_HOME`.
- Checkpoint locations should be stable relative paths under that root.
- The shipped repo should not promise checkpoint download, fetch, or registry tooling.
- Baseline flat checkpoints for `PushT` and `Cube` are explicit setup prerequisites.
- Probe training remains first-class, but the default reader-facing path consumes named probe checkpoints rather than requiring retraining.

### Outputs

- Checkpoint-producing specs write to:
  - `STABLEWM_HOME/runs/<name>/...`
- Derived outputs write to:
  - `STABLEWM_HOME/repro/<name>/...`

This split is part of the canonical contract and should be enforced consistently.

## Canonical Setup Decisions

### Environment

- Canonical install path is conda-based.
- `environment.yml` is the Tier 0 / developer baseline.
- `environment-gpu.yml` is the Tier 1 / Tier 2 base.
- Environments should be pinned for the supported reproduction platform.
- The formal reproducibility target is one canonical Linux `x86_64` CUDA platform.
- macOS or ad hoc local setups are best-effort developer environments, not part of the formal guarantee.

### Datasets

- One canonical dataset setup path should exist.
- Default paper-required datasets:
  - `pusht`
  - `cube`
- Extra datasets may remain optionally supported internally, but they are not part of the public paper path.

### Validation

The shipped repo should expose first-class validation commands for:

- environment sanity
- baseline integrity
- dataset presence
- checkpoint presence
- combined preflight

## Canonical Experiment Model

### Spec Formats

- Matrix inputs remain `CSV`.
- Non-matrix first-class specs use `YAML`.
- Generated summaries/manifests use `CSV` or `TSV` depending on the artifact.

### Registry Shape

The experiment tree should lead with operation type:

- `h_le_wm/experiments/train/...`
- `h_le_wm/experiments/matrix/...`
- `h_le_wm/experiments/diagnostic/...`
- `h_le_wm/experiments/probe/...`
- `h_le_wm/experiments/render/...`
- `h_le_wm/experiments/smoke/...`
- `h_le_wm/experiments/workflow/...`

All first-class specs should live under `h_le_wm/experiments/`.

### Runner Model

- One generic experiment runner should be the canonical backend.
- Thin wrapper scripts should provide the user-facing commands.
- A small machine-readable experiment index should describe the shipped first-class surface.
- Non-matrix specs should be small YAML overlays on top of reusable base configs.
- Matrix summarization is first-class and should produce canonical machine-readable summary artifacts.
- Paper render specs should depend on named upstream outputs, not hand-passed raw paths.

### Seed Policy

- Quantitative matrix experiments pin explicit seed lists and produce aggregated summaries.
- Qualitative diagnostics, probe renders, and story figures use one canonical pinned seed.

## Documentation Contract

The shipped docs should be minimal, normative, and directly actionable.

### Required Docs

- `README.md`
- `docs/install.md`
- `docs/datasets.md`
- `docs/reproduction.md`
- `docs/hardware.md`
- `docs/experiments.md`
- `docs/checkpoints.md`
- `docs/outputs.md`
- `docs/architecture.md`
- `docs/architecture_variants.md`

### Architecture Docs

The architecture docs should describe the supported mainline architecture, including:

- the base hierarchy
- the `third_party/` boundary
- the experiment system
- the checkpoint/output contract
- the paper reproduction flow

The variants doc should explicitly cover supported mainline branches:

- VQ macro-action path
- latent-action-dimension ablations
- empirical-macro / Samuele CEM constraint

It should not attempt to document every dormant branch.

## Folder Structure Target

```text
.
├── README.md
├── environment.yml
├── environment-gpu.yml
├── docs/
│   ├── install.md
│   ├── datasets.md
│   ├── reproduction.md
│   ├── hardware.md
│   ├── experiments.md
│   ├── checkpoints.md
│   ├── outputs.md
│   ├── architecture.md
│   └── architecture_variants.md
├── h_le_wm/
│   ├── config/
│   ├── experiments/
│   │   ├── index.yaml
│   │   ├── matrix/
│   │   └── specs/
│   ├── validation/
│   └── ...
├── scripts/
│   ├── setup_paper_datasets.sh
│   ├── setup_baseline_checkpoints.sh
│   ├── validate_preflight.sh
│   ├── run_pusht_smoke.sh
│   ├── run_pusht_baseline_matrix.sh
│   ├── run_pusht_hierarchical_matrix.sh
│   ├── run_cube_baseline_matrix.sh
│   ├── run_cube_hierarchical_matrix.sh
│   ├── run_pusht_offline_diagnostics.sh
│   ├── run_pusht_acting_diagnostics.sh
│   ├── train_pusht_probe_phase_a.sh
│   ├── train_pusht_probe_phase_b.sh
│   ├── render_pusht_paper_diagnostics.sh
│   ├── render_pusht_decoder_story_figures.sh
│   ├── render_pusht_story_figures.sh
│   └── run_paper_reproduction.sh
├── tests/
└── third_party/
```

## Priority Milestones

### 1. Canonical Surface and Output Contract

- create the `h_le_wm/` namespace
- establish the experiment registry and runner
- standardize `runs/` vs `repro/`
- make eval/output paths deterministic

This is the highest-priority milestone because every other reproducibility feature depends on it.

### 2. Setup and Preflight

- conda-based canonical install docs
- one paper-dataset setup path for `pusht,cube`
- explicit baseline checkpoint setup
- preflight validation commands

This is the second priority because a fresh-clone user needs a clean entry point before any experiment logic matters.

### 3. First-Class Matrix Reproduction

- baseline flat matrix for `PushT`
- hierarchical matrix for `PushT`
- baseline flat matrix for `Cube`
- hierarchical matrix for `Cube`
- generated machine-readable summaries for all matrices

This milestone covers the quantitative claim backbone of the paper.

### 4. PushT Diagnostics and Paper Renders

- offline diagnostics
- acting diagnostics
- generated summaries and deterministic output locations
- paper diagnostics render
- story-figure render chain

This milestone turns the paper narrative into a reproducible graph instead of a collection of ad hoc scripts.

### 5. Probe Pipeline

- Phase A probe spec
- Phase B probe spec
- checkpoint-consumed probe render path
- decoder-story figures wired into the canonical outputs

The probe is first-class, but it depends on the core training/checkpoint/output contract being stable.

### 6. PushT Smoke

- one self-contained GPU smoke train
- one self-contained GPU smoke eval
- `wandb` disabled by default and documented as such

The smoke path should be small, explicit, and account-free.

### 7. Docs Cleanup and Public Surface Pruning

- remove `TwoRooms` and `Reacher` from the shipped public surface
- remove notebooks from the formal paper path
- replace `jobs/`/`roadmap/` public guidance with normative docs
- keep only the supported mainline variants in architecture docs

This milestone makes the repo feel like a deliberate scientific artifact rather than an accumulated workspace.

## Success Criteria

The refactor is complete when:

- a fresh user can install the canonical environment
- the user can fetch the paper datasets
- the user can validate environment, datasets, baseline integrity, and checkpoints
- the user can run the official PushT smoke path on a supported GPU
- the user can run the first-class `PushT` and `Cube` matrix surfaces from named specs
- the user can run the PushT diagnostics, probe, and render surfaces from named specs
- all public first-class paths are script- and spec-driven
- the shipped docs explain both the architecture and the variants that matter for the paper
