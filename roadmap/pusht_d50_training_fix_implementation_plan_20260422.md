# PushT d=50 Top-Level Training Fix Implementation Plan

## Purpose

This note turns the earlier analysis into a concrete implementation plan focused only on training-side changes for the hierarchical model.

Scope constraints:

- keep the pretrained low level frozen by default
- do not retrain the baseline LeWM backbone from scratch
- do not increase the training trajectory span beyond the current `max_span=15`
- focus on changes that can improve the usefulness of the top level as a subgoal generator for PushT `d=50`

The main question is:

> Given that we keep the current short training windows, what is the highest-value way to make top-level training better aligned with how the hierarchy is actually used at evaluation time?

My answer is:

1. make waypoint semantics more consistent inside the existing 15-step training window
2. stop training the top level as a pure one-step teacher-forced predictor
3. add an explicit signal that the first predicted subgoal should be low-level reachable
4. put macro-action learning in the same representational language as the frozen low level

These changes are all top-level only. They do not require expanding trajectory length or retraining the low-level world model.

---

## Current Training Behavior

The current top-level training path is implemented in [hi_train.py](/gpfs/home2/scur0200/h-lewm/hi_train.py), with model structure in [hi_jepa.py](/gpfs/home2/scur0200/h-lewm/hi_jepa.py), waypoint sampling in [hi_waypoint_sampling.py](/gpfs/home2/scur0200/h-lewm/hi_waypoint_sampling.py), and default config in [config/train/hi_lewm.yaml](/gpfs/home2/scur0200/h-lewm/config/train/hi_lewm.yaml).

Operationally, training does the following:

1. sample waypoints from a short sequence window
2. encode waypoint observations into latent states
3. slice primitive actions between waypoint pairs
4. encode each action chunk into one macro-action latent
5. predict the next waypoint latent from the current waypoint latent plus that macro-action
6. optimize a one-step latent prediction loss

Under the current defaults:

- `wm.high_level.waypoints.num=5`
- `wm.high_level.waypoints.strategy=random_sorted`
- `wm.high_level.waypoints.max_span=15`
- `training.train_low_level=False`
- `loss.alpha=0.0`
- `loss.beta=1.0`
- `loss.sigreg.weight=0.0`

So the high-level model is being trained as a local supervised latent transition model, not as a planner-aware subgoal model.

That is acceptable for an initial prototype, but it is not closely aligned with evaluation, where:

- the planner performs repeated replanning
- only the first high-level subgoal is executed
- the low level must actually realize that first subgoal over a short primitive horizon

This mismatch is exactly where I think the training work should go.

---

## Guiding Principle

The top level does not need to become a perfect long-horizon planner during training.

It does need to become reliably good at one specific job:

> propose short-horizon latent subgoals that are stable under rollout and easy for the frozen low level to realize.

Given that goal, the most useful training improvements are not generic “more capacity” or “more compute” changes. They are changes that reduce the gap between:

- what the high level is optimized to do during training
- what the hierarchy actually needs from it at evaluation time

---

## Priority 1: Make Waypoint Semantics More Structured

### Problem

Inside the current fixed `max_span=15` window, the model is trained with `random_sorted` waypoint sampling and `num_waypoints=5`.

That means each training example can decompose the same local future span into many different gap patterns.

Example:

- one sample may use gaps like `1, 3, 4, 7`
- another may use `2, 2, 5, 6`
- another may use `1, 1, 1, 12`

All of these are legal under the current sampler. This creates two problems:

1. the macro-action encoder must absorb highly variable chunk durations and semantics
2. the high-level predictor sees inconsistent meanings for each position in the waypoint chain

So even before we discuss planning, the top level is learning from a noisy abstraction.

### Proposed Change

Switch the default training setup to fewer waypoints and more structured spacing, while keeping `max_span=15`.

Primary variant:

- `wm.high_level.waypoints.num=3`
- `wm.high_level.waypoints.strategy=fixed_stride`
- `wm.high_level.waypoints.stride=5`

Secondary ablation:

- same as above, but `stride=6`

This keeps the total span inside the same regime while making the high-level transition semantics much cleaner.

### Why This Helps

With `num_waypoints=3` and `stride=5`, each sample corresponds to two consistent macro transitions:

- current latent -> 5-step-later latent
- 5-step-later latent -> 10-step-later latent

This is far easier to interpret than a variable four-gap partition of a 15-step window.

The expected benefits are:

- more stable macro-action encoding
- more consistent predictor conditioning
- easier multi-step rollout training
- easier interpretation of what one top-level step means during evaluation

### Implementation

No code changes are required for the basic structured-sampling version, because [hi_waypoint_sampling.py](/gpfs/home2/scur0200/h-lewm/hi_waypoint_sampling.py) already supports `fixed_stride`.

The concrete changes are:

- add a dedicated training config override for structured PushT runs
- optionally make that structured setup the default for hierarchical PushT training

I would not remove the random sampler entirely. I would keep it available as an ablation path.

### Files To Change

- [config/train/hi_lewm.yaml](/gpfs/home2/scur0200/h-lewm/config/train/hi_lewm.yaml)
- optionally add a new config such as `config/train/experiment/hi_pusht_structured.yaml`

### Risk

The main risk is over-constraining the abstraction if the best subgoal spacing is not close to the fixed stride we choose.

That is why I would keep this as:

- one strong structured default
- one or two nearby stride ablations

But overall this is low-risk and high-value.

---

## Priority 2: Add Multi-Step High-Level Rollout Loss

### Problem

The current training loss in [hi_train.py](/gpfs/home2/scur0200/h-lewm/hi_train.py) is effectively:

`predict(z_t, macro_t) -> z_{t+1}`

with teacher-forced context latents.

That means the high-level predictor is only trained to do one-step supervised latent prediction. But evaluation uses the model inside a planner that depends on repeated latent rollouts and repeated replanning.

This is the classic failure mode:

- one-step loss looks reasonable
- autoregressive rollout drifts
- planned subgoals become less reliable as horizon or replanning depth increases

This is especially plausible for PushT `d=50`, where even a “local” hierarchy must be repeatedly correct.

### Proposed Change

Add a second top-level loss term that explicitly trains short autoregressive rollout over waypoint chains.

Concretely, after constructing:

- `z_context`
- `z_target`
- `macro_actions`

we train not only one-step prediction but also rollout prediction for 2 or 3 steps across the sampled waypoint sequence.

The model should use its own predicted latent as the next context state during the rollout branch.

### Suggested Objective

Keep the existing one-step loss, but add:

- `l2_rollout_loss`: MSE over rollout-predicted latents vs target latents across future waypoint positions

A practical combined objective:

`loss = beta_one_step * l2_pred_loss + beta_rollout * l2_rollout_loss + lambda_reach * reachability_loss + optional regularizers`

Recommended initial weighting:

- `beta_one_step = 1.0`
- `beta_rollout = 0.5`

Then tune based on optimization stability.

### Why This Helps

This directly attacks the training/evaluation mismatch.

The planner does not care whether the model is only locally accurate under teacher forcing. It cares whether the predicted latent chain remains useful under repeated use.

A rollout loss encourages:

- reduced compounding drift
- more stable latent dynamics under self-conditioning
- better compatibility with high-level planning, even if high-level horizon remains conservative

### Detailed Implementation Strategy

Add a helper in [hi_train.py](/gpfs/home2/scur0200/h-lewm/hi_train.py) that performs short autoregressive rollout on the waypoint chain.

High-level sketch:

1. start from `z_context[:, 0]`
2. apply `macro_actions[:, 0]` to predict next latent
3. feed that predicted latent back in with `macro_actions[:, 1]`
4. continue for a configured number of steps
5. compare each predicted latent with the corresponding `z_target`

Important implementation decision:

- use a dedicated rollout branch for training only
- do not replace the existing one-step supervision entirely

This gives stability while still teaching the model the local supervised target.

### Config Surface

Add something like:

```yaml
loss:
  rollout:
    enabled: True
    weight: 0.5
    steps: 2
```

If `steps=1`, this branch effectively collapses back to one-step.

### Files To Change

- [hi_train.py](/gpfs/home2/scur0200/h-lewm/hi_train.py)
- [config/train/hi_lewm.yaml](/gpfs/home2/scur0200/h-lewm/config/train/hi_lewm.yaml)

### Risk

The main risk is optimization instability if rollout supervision is weighted too heavily too early.

Mitigation:

- keep one-step loss active
- start with 2-step rollout only
- give rollout a moderate weight

This is the single most important training code change in the plan.

---

## Priority 3: Add First-Subgoal Reachability Loss Using The Frozen Low Level

### Problem

At evaluation time, only the first predicted high-level subgoal matters immediately.

In [hi_policy.py](/gpfs/home2/scur0200/h-lewm/hi_policy.py), the high-level planner rolls out a sequence and then extracts:

- `self._z_subgoal = pred[:, 0, 0, :]`

The low level then plans primitive actions toward that one subgoal.

But the current training objective does not explicitly care whether the predicted first subgoal is easy for the frozen low level to realize. It only cares whether the target latent transition matches the offline waypoint target.

This creates a direct objective mismatch:

- training says “predict the next latent well”
- execution needs “predict a subgoal the low level can actually hit”

### Proposed Change

Use the frozen low-level model as a reachability evaluator during top-level training.

The idea is not to train the low level. The idea is to use its dynamics to score whether a predicted first subgoal is feasible under a short primitive action budget.

### Concrete Loss Options

I would implement these in increasing complexity order.

#### Option A: Latent Reachability Consistency

For the first macro transition:

1. predict `z_subgoal_pred` with the top level
2. roll the frozen low-level model forward using the real primitive action chunk between those two waypoints
3. compare the resulting low-level latent `z_low_rollout` to `z_subgoal_pred`

Loss:

`reachability_loss = || z_low_rollout - z_subgoal_pred ||^2`

This asks the top level to produce subgoals consistent with what the frozen low level believes is reachable under the corresponding action chunk.

This is the simplest and safest version.

#### Option B: Targeted Reachability Margin

Use both:

- predicted subgoal
- true waypoint target

and penalize the predicted subgoal when it is farther from the low-level rollout than the true target is.

This is more comparative but less direct.

#### Option C: Low-Level Inner Optimization

This would approximate evaluation more closely by solving a short low-level planning problem inside training.

I do not recommend starting here. It is too expensive and too operationally complex for the first pass.

### Recommended Starting Point

Implement Option A first.

It gives a direct training signal with manageable complexity and no nested planner in the training loop.

### Why This Helps

This is the cleanest way to inject execution awareness into top-level training without unfreezing the low level.

The top level should stop learning only “what the offline waypoint sequence did” and start learning:

> what kind of latent subgoal is compatible with the frozen primitive dynamics model we will actually use at test time

That is the exact alignment we currently lack.

### Detailed Implementation Strategy

In [hi_train.py](/gpfs/home2/scur0200/h-lewm/hi_train.py):

1. isolate the first sampled waypoint pair
2. obtain the corresponding primitive action chunk
3. encode actions with the frozen low-level action encoder if needed
4. roll the low-level predictor from the first context latent over that chunk
5. compare the low-level final latent to the top-level predicted first subgoal

Important detail:

The low-level model in training currently uses grouped action encoding only when `train_low_level=True`, but we do not want to turn low-level training on globally just to compute this auxiliary loss.

So this change likely needs a separate helper path that:

- always computes the low-level action embeddings required for the reachability branch
- does so under frozen parameters
- does not activate the existing low-level training loss unless requested

### Config Surface

```yaml
loss:
  reachability:
    enabled: True
    weight: 0.25
    mode: low_rollout_consistency
```

Initial weight should be conservative.

### Files To Change

- [hi_train.py](/gpfs/home2/scur0200/h-lewm/hi_train.py)
- possibly [hi_jepa.py](/gpfs/home2/scur0200/h-lewm/hi_jepa.py) for a helper method that rolls the low level from a single latent over an action chunk
- [config/train/hi_lewm.yaml](/gpfs/home2/scur0200/h-lewm/config/train/hi_lewm.yaml)

### Risk

The main risks are:

- extra training compute
- possible tension between imitating the dataset latent and matching low-level reachability

This is why I would:

- apply the reachability loss only to the first transition
- keep its weight below the one-step prediction weight initially

This is the second most important code change after rollout loss.

---

## Priority 4: Encode Macro-Actions From Frozen Low-Level Action Embeddings

### Problem

Right now the macro-action encoder consumes raw primitive action chunks directly.

That means the top level is learning its own action abstraction from scratch, even though the frozen low level already has an action encoder that maps primitive actions into the latent conditioning space used by the low-level predictor.

This is wasteful and may produce a mismatch:

- low level “understands” actions in one embedding language
- top level invents another macro language from raw controls

### Proposed Change

Refactor macro-action encoding so that the top level operates on sequences of frozen low-level action embeddings rather than raw primitive action vectors.

Pipeline:

1. primitive action chunk
2. frozen low-level `action_encoder`
3. sequence of low-level action embeddings
4. top-level temporal aggregation module
5. macro-action latent

The aggregation module can still be a small Transformer with CLS pooling or another simple temporal encoder.

### Why This Helps

This encourages the hierarchy to speak a more consistent internal language.

Potential benefits:

- better representational reuse from the pretrained low level
- easier conditioning for the high-level predictor
- cleaner bridge between macro actions and low-level execution
- reduced pressure on the macro encoder to rediscover action semantics from scratch

### Implementation Strategy

This is a moderate refactor.

The simplest version is:

- keep the existing `latent_action_encoder` module interface
- change its input from raw `action_chunks` to encoded low-level action embeddings
- if needed, add a small projection layer so dimensions match cleanly

There are two design choices:

#### Design A: Reuse `action_encoder` outputs directly

If the low-level action encoder already outputs the same embedding dimension expected by the macro encoder, this is the cleanest version.

#### Design B: Add a preprocessing adapter

If shape or framing assumptions differ, add a small adapter:

- `low_action_emb -> adapter -> latent_action_encoder`

I would prefer Design A unless the existing architecture forces otherwise.

### Config Surface

```yaml
latent_action_encoder:
  input_source: low_level_action_emb  # or raw_action
```

This lets us keep the current raw-action path as an ablation.

### Files To Change

- [hi_train.py](/gpfs/home2/scur0200/h-lewm/hi_train.py)
- [hi_jepa.py](/gpfs/home2/scur0200/h-lewm/hi_jepa.py)
- possibly the latent action encoder module implementation if it assumes raw action dimension specifically
- [config/train/hi_lewm.yaml](/gpfs/home2/scur0200/h-lewm/config/train/hi_lewm.yaml)

### Risk

The main risk is that the low-level action encoder may not preserve exactly the information the top level needs for macro aggregation.

That is a real possibility, which is why this should be configurable rather than a hard replacement immediately.

Still, I think this is a strong medium-priority cleanup.

---

## Priority 5: Add Latent Robustness During Top-Level Training

### Problem

Teacher-forced training gives the high-level predictor clean latent contexts. Evaluation does not. At test time, the planner and model operate on model-generated or planner-induced latent states.

Even if we add rollout loss, it is useful to make the model robust to small latent deviations.

### Proposed Change

Add one or both of:

- latent noise augmentation on high-level context latents
- scheduled sampling for the high-level rollout branch

### Variant A: Small Latent Noise

During training, perturb `z_context` by small Gaussian noise:

`z_context_noisy = z_context + sigma * eps`

This trains the predictor to remain stable near the data manifold rather than only exactly on it.

### Variant B: Scheduled Sampling

In the rollout branch, gradually mix:

- teacher-forced next latent
- model-predicted next latent

This is more complex, but directly addresses the train/test mismatch.

### Recommendation

Start with small latent noise first. It is simpler, easier to debug, and enough to test whether mild robustness helps.

### Config Surface

```yaml
training:
  latent_noise:
    enabled: True
    std: 0.01
```

### Files To Change

- [hi_train.py](/gpfs/home2/scur0200/h-lewm/hi_train.py)
- [config/train/hi_lewm.yaml](/gpfs/home2/scur0200/h-lewm/config/train/hi_lewm.yaml)

### Risk

Too much noise will just hurt one-step predictive accuracy.

So this should be:

- small
- optional
- used only after rollout loss is in place

---

## Priority 6: Tighten Macro-Latent Distribution During Training

### Problem

The high-level planner at evaluation time searches over continuous latent macro-actions. If the training-time encoded macro latents occupy a narrow or structured manifold, the top level benefits from preserving that structure.

Currently there is no explicit regularization on macro-action latent geometry beyond what the encoder learns implicitly.

### Proposed Change

Add optional regularization on macro-action latents.

Good candidates:

- norm penalty
- variance floor across the batch
- covariance regularization
- simple Gaussian moment matching

### Recommended Starting Point

A lightweight version:

- log macro latent mean and variance statistics
- optionally add a small norm penalty if latent magnitudes become unstable

I would not lead with an elaborate distribution-matching objective before seeing the simpler fixes.

### Why This Helps

This is mainly defensive:

- keeps macro latents numerically well behaved
- may reduce extreme codes that later invite planner exploitation

### Config Surface

```yaml
loss:
  macro_latent_reg:
    enabled: False
    weight: 0.0
    type: l2_norm
```

### Files To Change

- [hi_train.py](/gpfs/home2/scur0200/h-lewm/hi_train.py)
- [config/train/hi_lewm.yaml](/gpfs/home2/scur0200/h-lewm/config/train/hi_lewm.yaml)

### Risk

This is easy to overdo and distort the representation, so I see it as optional support work, not a core fix.

---

## Priority 7: Add Better Training Diagnostics Before And Alongside Fixes

### Problem

A lot of the current iteration loop is blind. We can train a new top level, see eval performance move, and still not know why.

### Proposed Change

Add training-time logging for:

- waypoint gap histograms
- first-step rollout error vs teacher-forced error
- autoregressive rollout error by step index
- low-level reachability loss values
- macro latent norm statistics
- distance between predicted first subgoal and true target latent

If we implement the low-level-action-embedding path, also log:

- norm statistics of low-level action embeddings before macro aggregation

### Why This Helps

These diagnostics make it possible to separate:

- better one-step fitting
- better rollout stability
- better reachability alignment

Without these metrics, changes in evaluation score will remain hard to interpret.

### Files To Change

- [hi_train.py](/gpfs/home2/scur0200/h-lewm/hi_train.py)
- WandB logging path already wired through config

### Risk

Low. This is support infrastructure and should be added alongside the main changes.

---

## Proposed Implementation Order

I would implement the work in the following sequence.

### Phase 1: Lowest-Risk Alignment Improvements

1. structured waypoint config inside `max_span=15`
2. rollout-loss plumbing
3. richer logging for one-step vs rollout behavior

Reason:

- minimal architectural disruption
- directly addresses the biggest mismatch
- easiest to debug

### Phase 2: Execution-Aware Supervision

4. first-subgoal reachability loss through frozen low-level rollout
5. logging for reachability metrics

Reason:

- this is the first place where training becomes explicitly aware of the downstream controller
- it should only be added once the basic rollout branch is working

### Phase 3: Representation Cleanup

6. macro encoding from frozen low-level action embeddings
7. optional latent robustness / regularization

Reason:

- these are meaningful but more refactor-heavy
- they are best evaluated after the core training objective is improved

---

## Exact Changes I Would Start With

If we want the tightest initial implementation scope with the highest probability of useful signal, I would begin with exactly these changes:

1. add a structured top-level training config:
   - `num_waypoints=3`
   - `strategy=fixed_stride`
   - `stride=5`
   - keep `max_span=15`

2. add `l2_rollout_loss`:
   - 2-step autoregressive rollout over waypoint chain
   - keep current one-step loss active

3. add training diagnostics:
   - one-step loss
   - rollout loss
   - rollout loss by step
   - subgoal distance metrics

4. add optional reachability auxiliary loss:
   - first transition only
   - compare top-level predicted first subgoal to frozen low-level rollout endpoint under the same primitive chunk

That is the cleanest first implementation package.

---

## Changes I Would Explicitly Avoid For Now

To keep the project focused, I would avoid the following in the first training patch series:

- unfreezing the encoder
- unfreezing the low-level predictor
- enabling full low-level training loss by default
- increasing `max_span`
- adding a nested low-level CEM inside training
- introducing many new regularizers at once
- increasing model depth before fixing objective mismatch

These either violate the current scope or make diagnosis much harder.

---

## Expected Outcomes

If these fixes help, I would expect to see improvements in this order:

1. lower autoregressive rollout error even when one-step loss changes only modestly
2. more stable first-subgoal predictions during evaluation
3. less sensitivity to high-level planning horizon and replanning details
4. better d50 behavior under conservative planner settings

If they do not help, the likely interpretations become much clearer:

- if rollout loss improves but evaluation does not, the problem is more on planning/search side
- if reachability loss helps but rollout loss does not, execution mismatch was dominant
- if neither helps, the abstraction itself may be too weak and we should revisit planner constraints or evaluation setup

---

## Recommendation

My recommendation is to implement the first patch series as:

- structured waypoint training within the existing `max_span=15`
- multi-step high-level rollout loss
- first-subgoal reachability auxiliary loss
- supporting diagnostics

This keeps the low level frozen, respects the fixed training span, and targets the most important mismatch between current training and d50 execution.

If you approve, I would implement this in two code patches:

1. structured waypoint config plus rollout loss plus diagnostics
2. reachability auxiliary loss and the low-level helper path needed to compute it cleanly
