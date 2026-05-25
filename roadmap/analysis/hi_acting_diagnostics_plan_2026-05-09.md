# Acting Diagnostics Plan

Date: 2026-05-09

## Goal

Build an acting-diagnostics suite that measures whether planned subgoals and low-level plans actually work in the real PushT environment, not just inside the learned model.

This suite should answer, in order:

1. Can the low level execute oracle intermediate subgoals online?
2. What is the model-vs-reality gap for low-level plans?
3. Are generated high-level subgoals time-valid intermediate waypoints?
4. Is online failure caused by subgoal churn, low-level model exploitation, or bad first subgoals?

## Priority 1: Oracle-Subgoal Acting

This is the first and most important test.

For `d50 hh2`, run online env execution with:

- start state = true dataset state at time `t`
- stage 1 subgoal = true dataset state/obs at `t + 25`
- final goal = true dataset goal at `t + 50`

Run:

- `oracle_hh2 + lh2 + lrh1`
- `oracle_hh2 + lh3 + lrh1`
- `oracle_hh2 + lh5 + lrh1`

Metrics:

- actual env terminal latent error to stage 1 subgoal
- actual env terminal latent error to final goal
- actual goal progress in env
- success/contact proxy if available

Decision rule:

- if oracle subgoals work, the high-level generator is the main problem
- if oracle subgoals fail, the low-level online executor is the main problem

## Priority 2: Low-Level Reality Gap

Add an online version of the current reachability test.

For each sampled start and target subgoal:

- plan with low-level CEM in the model
- record predicted terminal latent
- execute selected action block(s) in the env
- re-encode actual resulting observation

Log:

- `model_error = ||z_pred_end - z_subgoal||`
- `actual_error = ||z_env_after_exec - z_subgoal||`
- `reality_gap = actual_error - model_error`
- `goal_progress = ||z_start - z_goal|| - ||z_env_after_exec - z_goal||`

Run this for:

- real dataset subgoals at offsets `+2`, `+3`, `+5`
- `lh2`, `lh3`, `lh5`
- optionally `lh5/lrh5` after `lrh1`

Decision rule:

- if model error is low but actual error is high, low-level CEM is exploiting rollout error

## Priority 3: Generated-Subgoal Acting

Add the online acting version of generated-subgoal reachability.

For each sampled start:

- generate high-level subgoal(s)
- execute low-level planning toward each generated subgoal in the env
- re-encode actual achieved state

Log:

- predicted subgoal error
- actual subgoal error
- reality gap
- actual env progress
- nearest future offset of the generated subgoal
- expected future offset
- offset error

For `d50 hh2`, expected offsets are:

- step 1: `+5` tokens
- step 2: `+10` tokens

Decision rule:

- if step 1 is nearest to the wrong future time, the subgoal is latent-plausible but not a valid intermediate waypoint

## Priority 4: Online Eval Logging

Extend `hi_eval.py` logging during real hierarchical runs.

Per env step or per low-level execution block, log:

- env step
- high-level plan id
- stage/subgoal id
- distance current -> current subgoal
- distance current -> final goal
- distance previous subgoal -> new subgoal
- predicted low-level terminal error
- actual post-exec terminal error
- actual goal progress
- selected low-level action norm / OOD score
- selected high-level macro norm / OOD score

Compare at least:

- `hh1_lh2_lrh1`
- `hh2_lh2_lrh1`
- `hh2_lh5_lrh1`
- `hh2_lh5_lrh5`

Decision rule:

- this should expose whether failure is:
  - bad first subgoal
  - subgoal churn
  - low-level model exploitation
  - no real progress despite low latent cost

## Priority 5: Low-Level Action OOD

Add a low-level analogue of the macro manifold diagnostic.

For selected low-level action tokens, log:

- dataset action-token norm
- selected action-token norm
- elite action-token norm
- per-dim z-score
- Mahalanobis distance
- fraction outside dataset quantiles
- fraction outside env action bounds before clipping

Run on:

- `lh2`
- `lh3`
- `lh5`

Decision rule:

- if `lh5` is much more OOD than `lh2/lh3`, that explains why offline `lh5` looks good but online `lh5` fails

## Implementation Layout

Add a second suite parallel to the offline diagnostics:

- `scripts/diagnostics/hi_acting_diagnostics.py`
- `scripts/diagnostics/run_hi_acting_diagnostic.py`
- `jobs/eval/hi/acting_diagnostics/checkpoints_acting.txt`
- `jobs/eval/hi/acting_diagnostics/run_acting_diagnostics_matrix.sh`
- `jobs/eval/hi/acting_diagnostics/submit_acting_diagnostics_matrix.sh`

Keep the same matrix style as the current diagnostics.

## Phase 1 Sweep

Start small. Do not launch a huge matrix first.

Use only:

- `latent32 epoch 15`
- `hope2 epoch 15`

Run:

1. oracle subgoal acting, `d50`, `lh2/lh3/lh5`
2. low-level reality gap on real dataset subgoals
3. generated-subgoal acting, `d50 hh1` and `d50 hh2`
4. online logging for normal `hh1` and `hh2`

## Success Criteria

This phase is successful if it answers one of these cleanly:

- oracle works, generated fails:
  - high-level subgoal generation is the main problem
- oracle fails too:
  - low-level online execution / model exploitation is the main problem
- generated step 1 has large offset error:
  - the first high-level waypoint is not temporally valid
- online logs show large subgoal churn:
  - replanning dynamics are the main problem

## Recommendation

Do not train a new model before this phase is done.

The first implementation target should be:

1. oracle-subgoal acting
2. low-level reality-gap logging
3. generated-subgoal time-validity logging
