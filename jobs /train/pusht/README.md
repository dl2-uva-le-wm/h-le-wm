# PushT Train Jobs

This directory holds PushT training and benchmark jobs.

## Layout

- `base/`
  - `train.sh`: base hierarchical PushT P2 training run.
- `hope1/`
  - `train_hope1.sh`: scratch-node training variant using local TMPDIR reads. Supports `MAX_EPOCHS` and `LATENT_ACTION_DIM` environment overrides.
  - `train_hope1_smoke.sh`: fast smoke version of `train_hope1.sh`.
- `hope2/`
  - `train_hope2.sh`: dedicated scratch-node P2 run with defaults `MAX_EPOCHS=15` and `LATENT_ACTION_DIM=32`.
  - `train_latent_action_dim_8.sh`: later latent-dim-8 P2 variant. Despite the filename comments referring to `hope3`, it is grouped here with the follow-on frozen-P2 variants.
  - `train_latent_action_dim_8_stride_5_n4.sh`: latent-dim-8 P2 variant with `waypoints.num=4` and fixed waypoint stride `5`.
  - `train_latent_action_dim_32_stride_5_n4.sh`: clean fixed-stride-5 ablation on the stronger dim-32 P2 recipe, again with `waypoints.num=4` to keep the default span budget unchanged.
  - Matching local `.out` / `.err` files are kept beside the corresponding script.
- `joint/`
  - `train_joint_levels.sh`: scratch-node joint P1+P2 training run with per-epoch object and training-state checkpoints.
  - `train_joint_levels_resume.sh`: resume an existing joint P1+P2 run in place from its saved `..._weights.ckpt`, keeping the same run directory and W&B id.
  - Matching local `.out` / `.err` files are kept beside the corresponding script.
- `benchmarks/`
  - `benchmark.sh`: short P2 throughput benchmark.
  - `benchmark_ab_io.sh`: A/B benchmark for shared scratch vs node-local TMPDIR I/O.
  - `benchmark_cpu_optimization.sh`: single node-local benchmark path.

## Local artifacts

Generated runtime artifacts (Slurm outputs and benchmark logs) are intentionally untracked:

- `*.out`, `*.err`, `*.log`
- `out/` runtime logs created by benchmark scripts
- `old/` historical local log snapshots
