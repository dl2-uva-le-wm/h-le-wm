# Eval Configuration Guide

This is a simple guide to the parameters you can set for `hi_eval.py` runs.

It covers:

- the parameters you usually set from the Slurm job scripts
- the important fields in `config/eval/hi_pusht.yaml`
- a few safety notes from what we learned during recent runs

## 1. Parameters You Will Actually Change Most Often

These are the practical knobs exposed by the job scripts in `jobs/eval/hi/`.

### Run / checkpoint selection

| Parameter | What it means | Typical use |
| --- | --- | --- |
| `RUN_NAME` | Which training run to evaluate. | Change this when you want to test a different trained checkpoint family. |
| `CHECKPOINT_EPOCH` | Which checkpoint to use from that run. Use `latest` or an integer like `8`. | Use `latest` for the newest model, or a specific epoch for ablations and comparisons. |
| `CONFIG_NAME` | Which Hydra eval config to load. Usually `hi_pusht`. | Only change this when switching task/config file. |
| `STABLEWM_HOME` | Root folder where runs and datasets live on the cluster. | Usually leave this alone unless your data is stored elsewhere. |

### Evaluation setup

| Parameter | What it means | Increase it when... | Decrease it when... |
| --- | --- | --- | --- |
| `GOAL_OFFSET_STEPS` | How far in the future the goal is. `25` is the short setting, `50` is the medium setting. | You want a harder / longer planning problem. | You want a shorter / easier planning problem. |
| `EVAL_BUDGET` | Maximum number of env steps allowed to reach the goal. | You want to give the planner more time. | You want stricter / cheaper evaluation. |
| `EVAL_DEVICE` | Device used during eval: `cpu`, `cuda`, or sometimes `auto`. | Use `cuda` if you really want GPU eval. | Use `cpu` for `rome` jobs or simpler cluster scheduling. |
| `RESULT_FILENAME` | Name of the result text file written into the artifact directory. | Change only if you want a custom output name. | Usually leave alone. |
| `EVAL_SUBDIR` | Output folder under the run directory for this eval. | Change only if you want a custom artifact folder name. | Usually leave alone. |

### High-level planner

| Parameter | What it means | Practical intuition |
| --- | --- | --- |
| `HIGH_HORIZON` | Number of high-level macro actions planned ahead. | Bigger = farther lookahead, but harder optimization. |
| `HIGH_ACTION_BLOCK` | How many times each high-level macro action is repeated. | Usually keep `1`. |
| `HIGH_RECEDING_HORIZON` | How much of the high-level plan you keep before replanning. | Usually keep `1`. |
| `HIGH_REPLAN_INTERVAL` | After how many env steps the high-level subgoal is replanned. | Smaller = more reactive, larger = more stable/cheaper. |
| `HIGH_NUM_SAMPLES` | Number of CEM samples for the high-level planner. | Bigger = broader search, more compute. |
| `HIGH_N_STEPS` | Number of CEM optimization iterations at high level. | Bigger = more refinement, more compute. |
| `HIGH_TOPK` | How many best candidates survive each CEM round at high level. | Lower = greedier search, higher = broader search. |

### Low-level planner

| Parameter | What it means | Practical intuition |
| --- | --- | --- |
| `LOW_HORIZON` | Number of low-level grouped actions planned ahead. | This was the strongest knob in your recent results. Lowering it from `5` to `2` helped a lot. |
| `LOW_ACTION_BLOCK` | Number of primitive env steps bundled into one low-level grouped action. | Important: for this checkpoint, keep this at `5`. |
| `LOW_RECEDING_HORIZON` | How much of the low-level plan you keep before replanning. | Usually keep `1`. |
| `LOW_NUM_SAMPLES` | Number of CEM samples for the low-level planner. | Bigger = broader search, more compute. |
| `LOW_N_STEPS` | Number of CEM optimization iterations at low level. | Bigger = more refinement, more compute. |
| `LOW_TOPK` | How many best low-level candidates survive each CEM round. | Larger often makes search less brittle, but too large can reduce focus. |

## 2. Important Safety Notes

### `LOW_ACTION_BLOCK`

This is the main dangerous parameter.

- For the current checkpoint family, `LOW_ACTION_BLOCK=5` is safe.
- Changing it to `3` caused a channel mismatch crash because the low-level action encoder expects grouped actions with size `10 = 5 * 2`.
- So for this checkpoint, treat `LOW_ACTION_BLOCK` as fixed unless you also change the model/checkpoint to match.

### `HIGH_ACTION_BLOCK`

- In your current setup this is effectively fixed at `1`.
- You can expose it, but it is not the first thing to tune.

### `RECEDING_HORIZON`

- In both high and low planners, this is usually `1`.
- It means “execute one grouped action chunk, then replan.”
- Most of your behavior differences came from `LOW_HORIZON` and `HIGH_REPLAN_INTERVAL`, not from changing receding horizon.

## 3. Base Config Fields In `config/eval/hi_pusht.yaml`

These fields live in [config/eval/hi_pusht.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml). Some are exposed by job scripts, some are only overridden through Hydra.

### Top-level fields

| Field | Meaning | Recommendation |
| --- | --- | --- |
| `cache_dir` | Where cached dataset files are read from. | Usually leave `null`. |
| `seed` | Random seed for eval sampling and solvers. | Change this when you want robustness / repeatability checks. |
| `policy` | Which policy/checkpoint to load. | Usually set indirectly via `RUN_NAME` + `CHECKPOINT_EPOCH`. |
| `device` | Runtime device: `auto`, `cpu`, or `cuda`. | Usually controlled by `EVAL_DEVICE`. |

### `world.*`

| Field | Meaning | Recommendation |
| --- | --- | --- |
| `world.env_name` | Which environment is evaluated. | Usually do not change unless changing task. |
| `world.num_envs` | Number of vectorized envs. | Derived from `eval.num_eval`; usually leave alone. |
| `world.max_episode_steps` | Hard episode cutoff. | It is set automatically in `hi_eval.py` to `2 * eval_budget`. |
| `world.history_size` | Observation history length. | Usually leave alone. |
| `world.frame_skip` | Environment frame skip. | Usually leave alone. |

### `dataset.*`

| Field | Meaning | Recommendation |
| --- | --- | --- |
| `dataset.stats` | Dataset name used for normalization stats. | Usually leave tied to `eval.dataset_name`. |
| `dataset.keys_to_cache` | Which dataset columns are cached. | Usually leave alone. |

### `eval.*`

| Field | Meaning | Recommendation |
| --- | --- | --- |
| `eval.num_eval` | Number of evaluation episodes / starts. | Increase for more stable estimates; decrease for faster experiments. |
| `eval.goal_offset_steps` | Goal distance in steps. | Main task-difficulty setting. |
| `eval.eval_budget` | Max allowed env steps. | Main “how much time do we give the planner?” setting. |
| `eval.img_size` | Image size after preprocessing. | Usually leave alone. |
| `eval.dataset_name` | Dataset to replay from. | Change only if switching dataset. |
| `eval.callables` | How initial state and goal state are injected into the env. | Usually leave alone. |

### `output.*`

| Field | Meaning | Recommendation |
| --- | --- | --- |
| `output.filename` | Results text filename. | Usually auto-generated by the script. |
| `output.subdir` | Artifact subdirectory. | Usually auto-generated by the script. |

## 4. Planner Config Structure

There are two planning modes.

### `planning.mode=hierarchical`

This uses:

- `planning.high.*` for the macro planner
- `planning.low.*` for the primitive/grouped-action planner

This is the mode you have been tuning most.

### `planning.mode=flat`

This ignores `planning.high.*` and `planning.low.*`, and instead uses:

- `solver.*`
- `plan_config.*`

That is the simple non-hierarchical planner.

## 5. Flat Planner Parameters

These matter only when `planning.mode=flat`.

| Field | Meaning | Recommendation |
| --- | --- | --- |
| `solver.num_samples` | Number of CEM samples in flat planning. | Bigger = broader search, more compute. |
| `solver.n_steps` | Number of CEM optimization iterations. | Bigger = more refinement, more compute. |
| `solver.topk` | Number of survivors per CEM round. | Tunes greediness vs diversity. |
| `solver.device` | Device for flat planner. | Usually same as global device. |
| `plan_config.horizon` | Number of grouped flat actions planned. | Bigger = farther lookahead, harder search. |
| `plan_config.receding_horizon` | How much of the plan you keep before replanning. | Usually `1`. |
| `plan_config.action_block` | Primitive env steps per grouped flat action. | For this task/checkpoint family, `5` is the normal setting. |

## 6. Hierarchical Planner Parameters

### `planning.high.solver.*`

| Field | Meaning |
| --- | --- |
| `planning.high.solver.num_samples` | High-level CEM sample count. |
| `planning.high.solver.n_steps` | High-level CEM iteration count. |
| `planning.high.solver.topk` | High-level survivor count. |
| `planning.high.solver.device` | Device for high-level CEM. |
| `planning.high.solver.batch_size` | Solver batch size. Usually leave alone. |
| `planning.high.solver.var_scale` | Initial search variance scale. Usually leave alone unless debugging CEM behavior. |
| `planning.high.solver.seed` | Solver seed. Usually inherited from `seed`. |

### `planning.high.plan_config.*`

| Field | Meaning |
| --- | --- |
| `planning.high.plan_config.horizon` | Number of macro actions planned ahead. |
| `planning.high.plan_config.receding_horizon` | Number of macro actions kept before replanning. |
| `planning.high.plan_config.action_block` | Repetitions per macro action. |

### `planning.high.replan_interval`

This is how often the high-level subgoal is refreshed in environment steps.

- Smaller = more reactive
- Larger = more stable and cheaper

### `planning.high.latent_prior.*`

These control the bounds used for the high-level latent action search.

| Field | Meaning | Recommendation |
| --- | --- | --- |
| `enabled` | Whether to estimate latent bounds from data. | Usually leave `True`. |
| `num_chunks` | Number of dataset chunks used to estimate bounds. | More chunks = more stable estimate, more startup cost. |
| `min_chunks_for_stats` | Minimum chunk count before trusting the estimate. | Usually leave alone. |
| `chunk_len` | Chunk length used during latent-bound calibration. | Usually leave alone. |
| `lower_q` / `upper_q` | Percentile range used to define bounds. | Controls how conservative the latent range is. |
| `margin_ratio` | Extra margin added around the percentile range. | Bigger = looser search bounds. |
| `clamp_abs` | Hard cap on latent bound magnitude. | Usually leave alone. |
| `fallback_abs` | Default absolute bound if calibration fails. | Usually leave alone. |

### `planning.low.solver.*`

| Field | Meaning |
| --- | --- |
| `planning.low.solver.num_samples` | Low-level CEM sample count. |
| `planning.low.solver.n_steps` | Low-level CEM iteration count. |
| `planning.low.solver.topk` | Low-level survivor count. |
| `planning.low.solver.device` | Device for low-level CEM. |
| `planning.low.solver.batch_size` | Solver batch size. Usually leave alone. |
| `planning.low.solver.var_scale` | Initial search variance scale. Usually leave alone. |
| `planning.low.solver.seed` | Solver seed. Usually inherited from `seed`. |

### `planning.low.plan_config.*`

| Field | Meaning |
| --- | --- |
| `planning.low.plan_config.horizon` | Number of grouped low-level actions planned ahead. |
| `planning.low.plan_config.receding_horizon` | Number of grouped low-level actions kept before replanning. |
| `planning.low.plan_config.action_block` | Primitive env steps per grouped low-level action. |

## 7. Recommended “Start Here” Tuning Order

If your goal is better performance rather than a full scientific sweep, this is a reasonable order:

1. `LOW_HORIZON`
2. `LOW_TOPK`
3. `LOW_NUM_SAMPLES` and `LOW_N_STEPS`
4. `HIGH_REPLAN_INTERVAL`
5. `EVAL_BUDGET`
6. `HIGH_HORIZON`

## 8. Practical Defaults For Your Current Checkpoint Family

For the current `hope1` checkpoint family, these are the safe defaults:

- `planning.mode=hierarchical`
- `EVAL_DEVICE=cpu` for `rome` jobs
- `HIGH_HORIZON=1`
- `HIGH_ACTION_BLOCK=1`
- `HIGH_RECEDING_HORIZON=1`
- `HIGH_REPLAN_INTERVAL=5`
- `LOW_ACTION_BLOCK=5`
- `LOW_RECEDING_HORIZON=1`

And these are the most promising tunable performance knobs:

- `LOW_HORIZON`
- `LOW_TOPK`
- `LOW_NUM_SAMPLES`
- `LOW_N_STEPS`
- `HIGH_REPLAN_INTERVAL`

If you are unsure what to change next, change one of those rather than the deeper config fields.
