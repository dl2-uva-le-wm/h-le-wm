# Hi Eval Jobs

Hierarchical PushT evaluation jobs, organized by goal offset and experiment type.

## Layout

- `d25/`: short-horizon eval jobs (`eval.goal_offset_steps=25`).
- `d50/`: medium-horizon eval jobs (`eval.goal_offset_steps=50`), including sweep submit helpers.
- `matrix/`: array-driven eval suites with checkpoint lists and hardcoded config matrices.
- `old_slurms/`: archived historical run outputs grouped by offset (`d25/`, `d50/`).
- `EVAL_CONFIG_GUIDE.md`: notes on key eval config knobs.
- `PLANNING_HPARAM_RESULTS.md`: planning hyperparameter observations.

## d25 scripts

- `joint_22368719_epoch25_d25_eval.sh`
- `hope3_ep13_d25_eval.sh`
- `stride5n4_ep15_d25_eval.sh`
- `d25_hierarchical_default_eval.sh`
- `d25_flat_default_eval.sh`
- `d25_hierarchical_soft_eval.sh`
- `d25_hierarchical_soft_low_budget_eval.sh`
- `d25_hierarchical_soft_low_horizon_base_eval.sh`
- `d25_hierarchical_soft_low_h1_block5_eval.sh`
- `d25_hierarchical_soft_low_h2_block5_eval.sh`
- `d25_hierarchical_soft_low_h2_searchboost_eval.sh`
- `d25_hierarchical_soft_low_h2_replan3_eval.sh`
- `d25_legacy_l2_policy_eval.sh`
- `d25_legacy_l2_policy_eval_minimal.sh`
- `d25/legacy/`: historical wrapper scripts retained for reference.

## d50 scripts

- `joint_22368719_epoch25_d50_eval.sh`
- `hope3_ep13_d50_eval.sh`
- `stride5n4_ep15_d50_eval.sh`
- `d50_hierarchical_default_eval.sh`
- `d50_hierarchical_soft_low_h2_eval.sh`
- `d50_hierarchical_soft_low_h2_paper_scaled_eval.sh`
- `submit_d50_cpu_overnight_sweep.sh`
- `d50/legacy/`: historical wrapper scripts retained for reference.
- `d50/sweeps/`: per-sweep local runtime output directories.

## Matrix jobs

- `matrix/eval_fixed_stride_matrix.sh`
  - Array job over checkpoint rows from `matrix/checkpoints_fixed_stride.txt`
  - Hardcodes the current 7-way fixed-stride eval suite:
    - `d25, hh1, lh2, lrh1`
    - `d25, hh1, lh5, lrh1`
    - `d25, hh2, lh2, lrh1`
    - `d50, hh1, lh2, lrh1`
    - `d50, hh2, lh2, lrh1`
    - `d50, hh2, lh5, lrh1`
    - `d50, hh2, lh5, lrh5`
- `matrix/submit_fixed_stride_matrix.sh`
  - Convenience submit helper for the fixed-stride matrix
  - Submits one 7-task array per checkpoint row automatically
  - Writes Slurm `.out/.err` files under `matrix/logs/<RUN_NAME>/`

## Artifact policy

Slurm/runtime artifacts (`*.out`, `*.err`, `*.log`, `submitted_jobs.tsv`) are intentionally
ignored by Git via `jobs/.gitignore` to keep this folder transition-safe for branch moves.

## Diagnostics

- `macro_action_manifold_cpu.sh`: CPU Slurm job for the macro-action manifold diagnostic
  (`scripts/test_macro_action_manifold.py`), comparing:
  - true dataset macro-actions vs one-step high-level prediction error
  - CEM macro-actions vs one-step error and off-manifold statistics
