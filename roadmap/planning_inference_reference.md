# Planning Inference Reference (Hierarchical + Flat)

This document explains how inference-time planning is configured in this repository, with explicit grounding in:

- HLWM paper: [HLWM_paper.pdf](/Users/niccolocaselli/Desktop/h-le-wm/roadmap/HLWM_paper.pdf)
- LeWM paper: [lwm_paper.pdf](/Users/niccolocaselli/Desktop/h-le-wm/roadmap/lwm_paper.pdf)

It also links choices to our local training/eval code:

- PushT eval config: [config/eval/hi_pusht.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml)
- Eval runner: [hi_eval.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_eval.py)
- Hierarchical policy and latent prior calibration: [hi_policy.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py)
- Hi-level training config: [config/train/hi_lewm.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/train/hi_lewm.yaml)
- PushT training data config: [config/train/data/hi_pusht.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/train/data/hi_pusht.yaml)

## 1) Quick start

Default hierarchical (current default):

```bash
python hi_eval.py --config-name=hi_pusht
```

Flat fallback:

```bash
python hi_eval.py --config-name=hi_pusht planning.mode=flat
```

Notes:
- `planning.mode=hierarchical` is the default in `hi_*` eval configs.
- In `flat` mode, top-level `solver` + `plan_config` are used.

## 2) Core meaning of horizon, action_block, receding_horizon

These are the most important planning knobs.

- `horizon`: how many decision blocks CEM optimizes in one solve call.
- `action_block`: how many consecutive env actions are packed inside one decision block.
- `receding_horizon`: how many optimized blocks are actually executed before replanning.

Derived quantities:

- optimized lookahead (env steps) = `horizon * action_block`
- executed per MPC cycle (env steps) = `receding_horizon * action_block`

Example (current low-level PushT):
- `horizon=5`, `action_block=5` -> CEM scores plans over `25` env steps.
- `receding_horizon=1` -> execute `5` env steps, then replan.

## 3) planning.mode

### `planning.mode=hierarchical` (default)

Implemented in [hi_eval.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_eval.py) + [hi_policy.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py):

1. Build two CEM solvers:
   - high-level CEM over latent macro-actions (`planning.high.*`)
   - low-level CEM over primitive/grouped actions (`planning.low.*`)
2. Optionally calibrate latent bounds once at eval startup (`planning.high.latent_prior.*`).
3. Every control step:
   - if macro replan boundary reached (`replan_interval`), solve high-level CEM toward goal latent
   - extract first latent subgoal
   - solve low-level CEM toward that subgoal
   - execute low-level receding-horizon chunk

Code path:
- mode switch + policy build: [hi_eval.py:93](/Users/niccolocaselli/Desktop/h-le-wm/hi_eval.py:93)
- latent prior calibration call: [hi_eval.py:116](/Users/niccolocaselli/Desktop/h-le-wm/hi_eval.py:116)
- policy high/low planning loop: [hi_policy.py:323](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py:323)

### `planning.mode=flat` (fallback)

In `flat` mode we instantiate `stable_worldmodel.policy.WorldModelPolicy` with top-level:
- `solver`
- `plan_config`

Code path:
- [hi_eval.py:95](/Users/niccolocaselli/Desktop/h-le-wm/hi_eval.py:95)

Important:
- Top-level `solver` + `plan_config` remain in config specifically for this fallback path.

## 4) PushT values and grounding (paper + our training)

Current PushT hierarchical values are in [config/eval/hi_pusht.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml).

### 4.1 Full hierarchical snapshot (`hi_pusht`)

| Config key | Value |
|---|---:|
| `planning.mode` | `hierarchical` |
| `planning.high.replan_interval` | `5` |
| `planning.high.solver.num_samples` | `900` |
| `planning.high.solver.n_steps` | `20` |
| `planning.high.solver.topk` | `10` |
| `planning.high.solver.var_scale` | `1.0` |
| `planning.high.plan_config.horizon` | `2` |
| `planning.high.plan_config.receding_horizon` | `1` |
| `planning.high.plan_config.action_block` | `1` |
| `planning.high.latent_prior.enabled` | `True` |
| `planning.high.latent_prior.num_chunks` | `2048` |
| `planning.high.latent_prior.min_chunks_for_stats` | `64` |
| `planning.high.latent_prior.chunk_len` | `5` |
| `planning.high.latent_prior.lower_q` | `5.0` |
| `planning.high.latent_prior.upper_q` | `95.0` |
| `planning.high.latent_prior.margin_ratio` | `0.05` |
| `planning.high.latent_prior.clamp_abs` | `3.0` |
| `planning.high.latent_prior.fallback_abs` | `1.0` |
| `planning.low.solver.num_samples` | `300` |
| `planning.low.solver.n_steps` | `30` |
| `planning.low.solver.topk` | `10` |
| `planning.low.solver.var_scale` | `1.0` |
| `planning.low.plan_config.horizon` | `5` |
| `planning.low.plan_config.receding_horizon` | `1` |
| `planning.low.plan_config.action_block` | `5` |

Source: [config/eval/hi_pusht.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml)

| Key | Current PushT value | Grounding in papers | Grounding in our code/training | Why this value makes sense |
|---|---:|---|---|---|
| `planning.high.plan_config.horizon` | 2 | HLWM Appendix C, Table 10, row `Push-T (d=25)`: high-level `pred H = 2` | [hi_pusht.yaml:38](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:38) | Matches HLWM high-level lookahead for PushT.
| `planning.high.solver.num_samples` | 900 | HLWM Appendix C, Table 10: high-level `#samples = 900` | [hi_pusht.yaml:31](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:31) | Directly aligned with HLWM PushT hierarchical setup.
| `planning.high.solver.n_steps` | 20 | HLWM Appendix C, Table 10: high-level `#iters = 20` | [hi_pusht.yaml:33](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:33) | Same optimization depth as paper row.
| `planning.high.solver.topk` | 10 | HLWM Appendix C, Table 10: high-level `#elites = 10` | [hi_pusht.yaml:34](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:34) | Same elite selection ratio used in paper setup.
| `planning.high.plan_config.receding_horizon` | 1 | Receding-horizon MPC in HLWM/LeWM methodology | [hi_pusht.yaml:39](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:39), [hi_policy.py:335](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py:335) | Use first high-level block now, keep/shift remainder with warm start.
| `planning.high.plan_config.action_block` | 1 | HLWM high-level plan is over latent macro-actions; with `H=2` this means two latent decisions per solve | [hi_pusht.yaml:40](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:40), [hi_jepa.py:447](/Users/niccolocaselli/Desktop/h-le-wm/hi_jepa.py:447) | Keeps latent search compact and directly aligned with `pred H=2`.
| `planning.high.replan_interval` | 5 | HLWM Appendix C, Table 10: `k = 5` for Push-T rows | [hi_pusht.yaml:26](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:26) | Replans macro-level every 5 env steps.
| `planning.low.plan_config.horizon` | 5 | HLWM Table 10 low-level `pred h = 5`; LeWM planning section uses horizon 5 for PushT | [hi_pusht.yaml:63](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:63) | Matches both HLWM hierarchical and LeWM flat horizon scale.
| `planning.low.solver.num_samples` | 300 | HLWM Table 10 low-level `#samples = 300`; LeWM planning uses 300 candidates per step | [hi_pusht.yaml:56](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:56) | Standard compute budget from LeWM/HLWM PushT.
| `planning.low.solver.n_steps` | 30 | HLWM Table 10 low-level `#iters = 30`; LeWM planning section: 30 iterations in PushT | [hi_pusht.yaml:58](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:58) | Same optimization depth used in PushT in both papers.
| `planning.low.solver.topk` | 10 | HLWM Table 10 low-level `#elites = 10` | [hi_pusht.yaml:59](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:59) | Follows HLWM hierarchical CEM row.
| `planning.low.plan_config.action_block` | 5 | LeWM: horizon 5 corresponds to 25 env steps because frame skip 5 | [hi_pusht.yaml:65](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:65), [hi_pusht train data:3](/Users/niccolocaselli/Desktop/h-le-wm/config/train/data/hi_pusht.yaml:3) | Training uses `frameskip=5`, so block-of-5 is temporally aligned.
| `planning.low.plan_config.receding_horizon` | 1 | MPC principle in both papers (receding-horizon replanning) | [hi_pusht.yaml:64](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:64), [hi_policy.py:376](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py:376) | Execute 1 low block (=5 env steps), then refresh plan.

Additional training consistency notes:
- Our high-level training uses variable waypoint gaps (`random_sorted`, `num=5`, `max_span=15`) in [config/train/hi_lewm.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/train/hi_lewm.yaml:55).
- Action chunks between waypoints are variable length and masked in [hi_train.py:419](/Users/niccolocaselli/Desktop/h-le-wm/hi_train.py:419).
- This is why latent macro-actions represent temporally extended behavior, not single primitive actions.

## 5) Exact config key reference

### `planning.high.*`

- `replan_interval`: number of env steps before forcing a new high-level solve.
- `solver`:
  - `_target_`: `stable_worldmodel.solver.CEMSolver`
  - `num_samples`: candidate plans per CEM iteration
  - `n_steps`: CEM iterations
  - `topk`: elite count
  - `var_scale`, `batch_size`, `device`, `seed`
- `plan_config`:
  - `horizon`: high-level decision blocks optimized each solve
  - `receding_horizon`: high-level blocks kept/executed from optimized plan
  - `action_block`: latent action grouping factor
- `latent_prior`:
  - `enabled`: turn latent-bound calibration on/off
  - `num_chunks`: number of sampled action chunks from dataset
  - `min_chunks_for_stats`: minimum latent samples needed, else fallback bounds
  - `chunk_len`: primitive actions per sampled chunk
  - `lower_q`, `upper_q`: percentile bounds per latent dimension
  - `margin_ratio`: expands percentile interval
  - `clamp_abs`: optional absolute clipping of bounds
  - `fallback_abs`: fallback symmetric box if calibration fails

Where implemented:
- calibration logic: [hi_policy.py:42](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py:42)
- calibration call frequency: once at policy build in [hi_eval.py:116](/Users/niccolocaselli/Desktop/h-le-wm/hi_eval.py:116)

### `planning.low.*`

- `solver`: same CEM fields as high-level, but in primitive/grouped action space.
- `plan_config`:
  - `horizon`: low-level blocks optimized each solve
  - `receding_horizon`: low-level blocks executed before replanning
  - `action_block`: primitive actions per block

Where used:
- low-level solve and action buffering: [hi_policy.py:351](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py:351)

### Other eval keys

- `world.*`
  - `env_name`, `num_envs`, `history_size`, `frame_skip`, optional env-specific keys (`task` in Reacher)
  - `max_episode_steps` is set at runtime to `2 * eval_budget` in [hi_eval.py:166](/Users/niccolocaselli/Desktop/h-le-wm/hi_eval.py:166)
- `dataset.*`
  - `stats` and `keys_to_cache` (columns used for normalizers and cached loading)
- `eval.*`
  - `num_eval`: number of start-goal eval rollouts
  - `goal_offset_steps`: goal sampled this many steps ahead in dataset trajectory
  - `eval_budget`: max executed actions per episode
  - `dataset_name`: dataset source
  - `callables`: env reset/state setter hooks
- `policy`
  - cost model checkpoint key for `swm.policy.AutoCostModel`
- `output.filename`
  - result file name appended under eval result path

## 6) Flat fallback keys (top-level)

These remain active only when `planning.mode=flat`:

- `solver` (top-level)
- `plan_config` (top-level)

Current PushT top-level values are in [config/eval/hi_pusht.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:67).

## 7) Paper citations used (exact locations)

1. HLWM hierarchical PushT CEM hyperparameters:
   - Appendix C, Table 10, row `Push-T (d = 25)` in [HLWM_paper.pdf](/Users/niccolocaselli/Desktop/h-le-wm/roadmap/HLWM_paper.pdf)
   - Values used there: high (`#elites=10`, `#iters=20`, `#samples=900`, `pred H=2`, `k=5`), low (`#elites=10`, `#iters=30`, `#samples=300`, `pred h=5`).
2. HLWM method statement for two temporal scales and receding-horizon MPC:
   - Main text Sec. 3 (hierarchical planning description), [HLWM_paper.pdf](/Users/niccolocaselli/Desktop/h-le-wm/roadmap/HLWM_paper.pdf)
3. LeWM planning solver defaults for PushT:
   - Appendix D, "Planning solver" in [lwm_paper.pdf](/Users/niccolocaselli/Desktop/h-le-wm/roadmap/lwm_paper.pdf)
   - CEM with `300` samples, up to `30` iterations, top `30` elites, horizon `5`.
4. LeWM frame-skip interpretation:
   - Appendix D implementation details: frame skip `5`, where horizon `5` corresponds to `25` env timesteps, [lwm_paper.pdf](/Users/niccolocaselli/Desktop/h-le-wm/roadmap/lwm_paper.pdf)
5. LeWM PushT eval protocol:
   - Appendix eval setup: PushT uses `eval_budget=50` and goals sampled `25` steps ahead, [lwm_paper.pdf](/Users/niccolocaselli/Desktop/h-le-wm/roadmap/lwm_paper.pdf)

## 8) Important practical note on latent prior calibration

Current calibration samples fixed-length chunks (`chunk_len=5`) before eval start, then builds a per-dimension latent Box for high-level CEM.

This is stable and practical, but training saw variable chunk lengths between sampled waypoints. So fixed `chunk_len=5` is a pragmatic approximation, not a perfect replay of train-time chunk-length distribution.

If needed, next improvement is variable-length calibration that reuses the same waypoint-gap distribution used in training.
