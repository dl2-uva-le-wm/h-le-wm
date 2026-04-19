# PushT `d=25` Failure Report for Hi-LeWM

## 0) Situation Summary

This document summarizes the current state of the Hi-LeWM Push-T effort so a new reader can understand what was attempted, what has been verified, what is still failing, and what should be fixed next.

### Project goal

The goal is to extend the original LeWM setup into a usable hierarchical method without losing short-horizon performance.

For Push-T, the intended success criteria are:

1. keep short-horizon performance at `d=25` at baseline level
2. improve over the flat baseline at longer horizon such as `d=50`

In practice, the intended unified behavior is:

- `d=25`: use the flat planner because the task is already reachable in a short horizon
- `d>=50`: use the hierarchical planner because longer-horizon mediation should help

### What has been done so far

We trained only the second level (`P2`) on top of the original pretrained LeWM world model instead of retraining the full stack from scratch.

For the run reported here:

- pretrained checkpoint: `pusht/lewm_object.ckpt`
- training script: `hi_train.py`
- run name: `hi_lewm_p2_train_hope1_21983875`
- training date from the logs: `2026-04-18`
- max epochs: `10`
- low-level model: frozen
- trainable parts: `p2_high_predictor`, `p2_latent_action_encoder`, `p2_high_pred_proj`
- total parameters: `30,550,958`
- trainable parameters: `12,516,480`

The run configuration explicitly trained the high-level objective only:

- `training.train_low_level=False`
- `loss.alpha=0.0`
- `loss.beta=1.0`

### What the training and validation say

The training run does **not** look catastrophically broken.

Evidence from the logs and exported curves:

- initial validation `l2_pred_loss`: `0.7214953899383545`
- final validation `l2_pred_loss`: `0.019426284357905388`
- best validation `l2_pred_loss` in the shown epoch summaries: about `0.019371245056390762`
- train and validation losses both decrease smoothly
- the run is numerically stable: no NaNs, no divergence, no late collapse
- the gradient check reports that all tracked trainable parameters received gradients on the first backward pass

Supporting exports:

- [train.csv](/Users/niccolocaselli/Downloads/train.csv)
- [validation.csv](/Users/niccolocaselli/Downloads/validation.csv)

The main training-side yellow flag is not collapse, but scale drift:

- `validate/macro_action_norm` rises during training from about `25.7` to about `29.7`
- this suggests that the learned high-level latent-action scale is moving during training, which can matter for planning and CEM calibration

### What seems to have worked

- the frozen low-level stack loaded correctly
- the intended P2 modules were trainable
- training made clear progress on the offline objective
- validation tracked training rather than exploding away from it
- `high_pred_proj` does not look trivially dead or disconnected

### What has not worked

The current method is still failing the practical acceptance test at short horizon.

- reported result: `success_rate = 14.0` (`7/50`) at `d=25`
- this is far below the original LeWM Push-T baseline and also below what a usable short-horizon non-regression path would require

So the current situation is:

- offline training looks healthy
- online planning performance at `d=25` is still unacceptable

### Current interpretation

Based on the evidence available right now, the most likely conclusion is:

1. the P2 training run is not obviously the main failure
2. the bridge from trained high-level latents to actual planning is where the main problems are
3. there are also concrete evaluation-path issues that make some current comparisons invalid

In other words, the present evidence points more toward:

- invalid short-horizon parity settings
- missing planner-switch logic
- calibration problems in the high-level latent prior
- hierarchy hurting a task where flat planning may already be enough

rather than toward:

- total training failure
- a dead `high_pred_proj`
- catastrophic optimization problems in the P2 run

This is why the rest of this report focuses mainly on root causes in validation and planning rather than treating the training run itself as obviously broken.

---

## 1) Problem Statement

The short-horizon non-regression gate is currently not satisfied.

- Reported result: `success_rate = 14.0` (`7/50`) at `d=25`
- Required outcome: baseline parity at `d=25`
- Why this matters: if `d=25` is badly regressed, the extension is not viable as a unified method

This document was updated after re-checking the paper baselines and cross-validating each suspected issue against the current codebase.

---

## 2) Paper Ground Truth

### Original LeWM paper

For the standard Push-T setup where the goal is reachable within 25 steps and the planning budget is 50 steps, LeWM reports:

- **LeWM**: `96.0 ± 2.83`

Source:
- [roadmap/lwm_paper.txt](/Users/niccolocaselli/Desktop/h-le-wm/roadmap/lwm_paper.txt:2434)

### HLWM paper

For Push-T with DINO-WM, the hierarchical paper reports:

- `d=25`: flat `84%`, hierarchical `89%`
- `d=50`: flat `55%`, hierarchical `78%`

Source:
- [roadmap/HLWM_paper.txt](/Users/niccolocaselli/Desktop/h-le-wm/roadmap/HLWM_paper.txt:617)

These are the relevant baseline anchors for the acceptance criteria:

1. `d=25` must achieve parity with the short-horizon baseline regime
2. `d=50` must improve over the flat baseline

---

## 3) Executive Diagnosis

After checking the code, the main problem is not a single modeling bug. It is a combination of:

1. A broken short-horizon parity path
2. A real design tradeoff where hierarchy can hurt `d=25`
3. A calibration bug in the high-level latent prior
4. Stale eval entrypoints that can fail before producing valid results

Some items from the earlier report were valid concerns, but not actual implementation bugs. Those have been downgraded or removed below.

---

## 4) Verified Issues

## Issue 1: The `d=25` flat fallback is not baseline-equivalent (Critical)

### What is true

The repo does have a flat fallback path, but it does **not** match baseline LeWM planning.

- Baseline Push-T eval config:
  - `horizon=5`
  - `receding_horizon=5`
  - `action_block=5`
  - [third_party/lewm/config/eval/pusht.yaml](/Users/niccolocaselli/Desktop/h-le-wm/third_party/lewm/config/eval/pusht.yaml:23)

- Current hierarchical eval config top-level flat fallback:
  - `horizon=1`
  - `receding_horizon=1`
  - `action_block=5`
  - [config/eval/hi_pusht.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:78)

- `hi_eval.py` really uses those top-level keys in flat mode:
  - [hi_eval.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_eval.py:101)

### Why it matters

Even if evaluation switches to `planning.mode=flat`, the current fallback still uses a much shorter MPC regime than baseline LeWM. That means the intended `d=25` parity check is invalid today.

### Conclusion

This is a real bug relative to the project acceptance criteria.

---

## Issue 2: Dynamic horizon-aware planner switching is not implemented (Critical)

### What is true

The intended unified method is:

- use flat planning for `d=25`
- use hierarchical planning for `d>=50`

But the code currently chooses planner mode only from static config:

- [hi_eval.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_eval.py:101)

There is no branch on:

- `eval.goal_offset_steps`
- planning horizon category
- `d=25` vs `d=50`

### Why it matters

The core project hypothesis has not been realized in code yet. So current results are not evaluating the intended unified method.

### Conclusion

This is a real missing implementation, not just a tuning issue.

---

## Issue 3: Hierarchical subgoal mediation is a real short-horizon risk (High)

### What is true

The hierarchical policy genuinely does:

1. plan at the high level toward the goal latent
2. roll out a high-level latent plan
3. take the first predicted latent as subgoal
4. make the low-level planner optimize toward that subgoal instead of the final goal

Relevant code:

- high-level solve: [hi_policy.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py:368)
- subgoal extraction from high rollout: [hi_policy.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py:399)
- low-level solve against `z_subgoal`: [hi_policy.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py:404)

### Why it matters

This is not a code defect by itself, but it is a plausible reason for a `d=25` regression. At short horizon, flat primitive planning may already be sufficient, and the extra subgoal layer can inject avoidable prediction error.

### Conclusion

Keep this as a valid root-cause hypothesis, but classify it as a design-level issue, not an implementation bug.

---

## Issue 4: The hierarchical low-level planner is weaker than the baseline planner at `d=25` (High)

### What is true

- Baseline CEM uses `topk=30`
  - [third_party/lewm/config/eval/solver/cem.yaml](/Users/niccolocaselli/Desktop/h-le-wm/third_party/lewm/config/eval/solver/cem.yaml:1)

- Hierarchical low-level Push-T config uses:
  - `topk=10`
  - `receding_horizon=1`
  - [config/eval/hi_pusht.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:51)

### Why it matters

At `d=25`, the extension is not being compared against baseline with equivalent low-level control strength. This is part of the parity failure.

### Conclusion

This is a real configuration mismatch and should stay in the report.

---

## Issue 5: High-level latent-prior calibration samples across episode boundaries (High)

### What is true

`calibrate_latent_prior()` currently:

1. loads the full action table
2. removes NaN rows
3. samples contiguous chunks from the flattened action array
4. encodes those chunks into latent macro-actions

Code:

- action extraction and flatten-style sampling:
  - [hi_policy.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py:92)
  - [hi_policy.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py:146)
  - [hi_policy.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_policy.py:165)

What it does **not** do:

- respect episode IDs
- prevent a sampled chunk from crossing trajectory boundaries

### Why it matters

This is stronger than the earlier “fixed `chunk_len=5` may mismatch training” hypothesis. The current code can build latent bounds from invalid action chunks that never occurred in the data distribution.

That can distort the high-level CEM search box and hurt planning quality.

### Conclusion

This is a real implementation bug and a new concrete discovery.

---

## Issue 6: Eval scripts and docs are stale and can fail before producing valid measurements (High)

### What is true

Several eval entrypoints still pass the removed Hydra key `wm.num_levels=2`.

Examples:

- [jobs/eval/hi/eval_hope1_medium.sh](/Users/niccolocaselli/Desktop/h-le-wm/jobs/eval/hi/eval_hope1_medium.sh:183)
- [jobs/eval/hi/pusht_eval_l2_d25.sh](/Users/niccolocaselli/Desktop/h-le-wm/jobs/eval/hi/pusht_eval_l2_d25.sh:107)
- [jobs/eval/hi/pusht_eval_l2_d25_simple.sh](/Users/niccolocaselli/Desktop/h-le-wm/jobs/eval/hi/pusht_eval_l2_d25_simple.sh:30)

The saved stderr logs show Hydra rejecting that override:

- [jobs/eval/hi/eval_hope1_short_21994341.err](/Users/niccolocaselli/Desktop/h-le-wm/jobs/eval/hi/eval_hope1_short_21994341.err:16)
- [jobs/eval/hi/eval_hope1_medium_21994342.err](/Users/niccolocaselli/Desktop/h-le-wm/jobs/eval/hi/eval_hope1_medium_21994342.err:16)

The README is stale in the same way:

- [README.md](/Users/niccolocaselli/Desktop/h-le-wm/README.md:74)
- [README.md](/Users/niccolocaselli/Desktop/h-le-wm/README.md:108)

### Why it matters

This means some reported runs may not have executed the intended evaluation path at all. At minimum, the repo’s current documented eval path is unreliable.

### Conclusion

This is a real repo health issue and a new concrete discovery.

---

## 5) Items From the Earlier Report That Are Not Bugs

## Non-Issue A: The eval sampling mismatch claim is outdated

The earlier report said baseline and hierarchical evaluators use different start-index sampling logic. That is no longer true in a meaningful way for the current hierarchical path.

- hierarchical helper now samples from the full valid index population:
  - [hi_eval.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_eval.py:27)
- there is a dedicated regression test:
  - [tests/test_hi_eval_sampling.py](/Users/niccolocaselli/Desktop/h-le-wm/tests/test_hi_eval_sampling.py:24)

This should be removed as an active root cause.

---

## Non-Issue B: Objective-performance misalignment is a risk, not an implementation defect

The P2 training objective is exactly what the config says:

- `training.train_low_level=False`
- `loss.alpha=0.0`
- `loss.beta=1.0`
- [config/train/hi_lewm.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/train/hi_lewm.yaml:80)

And the loss is wired consistently in training:

- [hi_train.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_train.py:450)

This may still be a poor experimental choice, but it is not a broken code path.

---

## Non-Issue C: `max_epochs=10` is not a bug

The training job does cap the run at 10 epochs:

- [jobs/train/pusht/train_hope1.sh](/Users/niccolocaselli/Desktop/h-le-wm/jobs/train/pusht/train_hope1.sh:151)

That may be too short, but it is a run decision, not an implementation failure.

---

## Non-Issue D: Fixed `chunk_len=5` alone is not the main bug

Using a fixed calibration chunk length is an approximation:

- [config/eval/hi_pusht.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/eval/hi_pusht.yaml:41)

Training does use variable macro-action lengths:

- [config/train/hi_lewm.yaml](/Users/niccolocaselli/Desktop/h-le-wm/config/train/hi_lewm.yaml:41)
- [hi_train.py](/Users/niccolocaselli/Desktop/h-le-wm/hi_train.py:417)

But the more important concrete problem is that calibration ignores episode boundaries. The report should emphasize that stronger finding instead of over-claiming that fixed length by itself is the bug.

---

## Non-Issue E: The runtime-warning claim is not substantiated by the checked local logs

I did not find supporting evidence for the previously cited observation-space warning in the local eval artifacts reviewed for this report.

This item should be removed unless new logs are added.

---

## 6) Updated Remediation Order

## Phase A: Restore a valid non-regression path

1. Fix stale eval scripts and README examples that still pass removed keys
2. Make flat fallback truly baseline-equivalent for Push-T `d=25`
3. Re-run `d=25` flat parity using the same sampled starts as baseline

## Phase B: Implement the intended unified method

1. Add horizon-aware switching in eval:
   - `d=25` -> flat
   - `d>=50` -> hierarchical
2. Report both settings as one method with dynamic planner selection

## Phase C: Fix the high-level calibration bug

1. Rebuild latent-prior calibration so chunk sampling respects episode boundaries
2. Then decide whether variable-gap calibration is needed on top

## Phase D: Re-evaluate hierarchy itself

1. `d=25` flat parity mode
2. `d=25` hierarchical mode
3. `d=50` hierarchical mode

This cleanly separates:

- parity failure
- hierarchy-induced short-horizon degradation
- long-horizon value-add

---

## 7) Final Assessment

The original conclusion still stands: `14%` at `d=25` is unacceptable.

But the report should no longer frame the situation as only a vague “stacked mismatch.” The code review shows several specific, actionable problems:

1. the current flat fallback is not baseline-equivalent
2. the intended dynamic planner switch is not implemented
3. high-level latent-prior calibration is sampling invalid cross-episode chunks
4. multiple eval scripts and docs are stale enough to fail outright

Those are the highest-signal fixes to make before interpreting any new Push-T results.
