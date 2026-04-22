# PushT Train Jobs

This directory holds PushT training and benchmark jobs.

## Main scripts

- `train.sh`: base hierarchical PushT P2 training run.
- `train_rollout_10ep.sh`: hierarchical PushT P2 run with rollout loss enabled (`10` epochs default).
- `train_hope1.sh`: scratch-node training variant using local TMPDIR reads.
- `train_hope1_smoke.sh`: fast smoke version of `train_hope1.sh`.
- `benchmark.sh`: short P2 throughput benchmark.
- `benchmark_ab_io.sh`: A/B benchmark (shared scratch vs node-local TMPDIR I/O).
- `benchmark_cpu_optimization.sh`: single node-local benchmark path.

## Local artifacts

Generated runtime artifacts (Slurm outputs and benchmark logs) are intentionally untracked:

- `*.out`, `*.err`, `*.log`
- `out/` runtime logs created by benchmark scripts
- `old/` historical local log snapshots
