# Hi Eval Jobs

Hierarchical PushT evaluation jobs, organized by goal offset and experiment type.

## Layout

- `d25/`: short-horizon eval jobs (`eval.goal_offset_steps=25`).
- `d50/`: medium-horizon eval jobs (`eval.goal_offset_steps=50`), including sweep submit helpers.
- `old_slurms/`: archived historical run outputs grouped by offset (`d25/`, `d50/`).
- `EVAL_CONFIG_GUIDE.md`: notes on key eval config knobs.
- `PLANNING_HPARAM_RESULTS.md`: planning hyperparameter observations.

## d25 scripts

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

- `d50_hierarchical_default_eval.sh`
- `d50_hierarchical_soft_low_h2_eval.sh`
- `d50_hierarchical_soft_low_h2_paper_scaled_eval.sh`
- `submit_d50_cpu_overnight_sweep.sh`
- `d50/legacy/`: historical wrapper scripts retained for reference.
- `d50/sweeps/`: per-sweep local runtime output directories.

## Artifact policy

Slurm/runtime artifacts (`*.out`, `*.err`, `*.log`, `submitted_jobs.tsv`) are intentionally
ignored by Git via `jobs/.gitignore` to keep this folder transition-safe for branch moves.

## Diagnostics

- `macro_action_manifold_cpu.sh`: CPU Slurm job for the macro-action manifold diagnostic
  (`scripts/test_macro_action_manifold.py`), comparing:
  - true dataset macro-actions vs one-step high-level prediction error
  - CEM macro-actions vs one-step error and off-manifold statistics
