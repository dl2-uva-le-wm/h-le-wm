# Original Baseline Eval Jobs

This directory contains PushT evaluation jobs that run the original LeWM baseline
(`third_party/lewm/eval.py`) with different eval variants.

## Scripts

- `pusht_eval.sh`: baseline PushT eval.
- `pusht_eval_withmetrics.sh`: baseline eval + per-eval pass/fail manifest.
- `pusht_eval_withmetrics_budget.sh`: metric eval variant with larger eval budget.
- `pusht_eval_withmetrics_horizon.sh`: metric eval variant with longer planning horizon.

## Runtime outputs

Slurm outputs are written under `out/` by these scripts.
Those `*.out`/`*.err` files are intentionally ignored by Git.
