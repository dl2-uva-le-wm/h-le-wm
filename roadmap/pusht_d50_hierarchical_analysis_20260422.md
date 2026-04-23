# PushT d=50 Hierarchical Analysis

## Scope

This note summarizes what the current hierarchical model actually is in this repo, why it likely underperforms on PushT `d=50`, and what I would try next.

The analysis is based on:

- training code in `hi_train.py`
- model code in `hi_jepa.py`, `hi_module.py`, `hi_policy.py`, `hi_waypoint_sampling.py`
- default configs in `config/train/hi_lewm.yaml`, `config/train/data/hi_pusht.yaml`, `config/eval/hi_pusht.yaml`
- packaged run scripts in `roadmap/jobs_github_ready_20260422.zip`
- baseline LeWM code in `third_party/lewm`
- existing planning notes in `jobs/eval/hi/PLANNING_HPARAM_RESULTS.md`

## What model we are actually running

This is not a jointly-trained hierarchical world model from scratch. It is a frozen-low-level plus trained-high-level hybrid.

### Low level

From `hi_train.py` and `hi_jepa.py`:

- `encoder`, `low_predictor`, `action_encoder`, `projector`, and `low_pred_proj` are loaded from a pretrained LeWM checkpoint.
- In the default PushT training job, all of those are frozen.
- So the low-level latent dynamics and primitive action embedding are inherited from the baseline LeWM.

Operationally, the low level is still the baseline LeWM one-step latent dynamics model:

- it predicts next latent from recent latent context plus primitive actions
- at eval time it is used inside CEM as the low-level planner cost model

### High level

The trainable hierarchical part is:

- `latent_action_encoder`: a small Transformer with CLS pooling that compresses a variable-length primitive action chunk into one latent macro-action
- `macro_to_condition`: maps macro-action latent to the predictor conditioning space
- `high_predictor`: another LeWM-style autoregressive predictor operating over waypoint latents
- `high_pred_proj`: a separate trainable projection head for the high-level predictor

Training in `hi_lejepa_forward()` is:

1. sample waypoint indices inside a short training window
2. encode waypoint frames to latents
3. slice the primitive actions between waypoint pairs
4. encode each action chunk into one macro-action latent
5. predict next waypoint latent from current waypoint latent plus macro-action
6. optimize only the high-level latent prediction loss by default

So the high level is trained as:

`(z_t, primitive actions between waypoints) -> z_next_waypoint`

not as:

- direct subgoal reachability model
- hierarchical imitation policy
- joint low/high planner-aware objective

### Planning at eval time

Hierarchical eval in `hi_policy.py` is two nested MPC loops:

- high-level CEM plans in macro-action latent space
- low-level CEM plans primitive actions toward the current high-level subgoal

Important detail:

- the high-level cost is only terminal latent distance to `z_goal`
- then only the first predicted subgoal `pred[:, 0, 0, :]` is passed to the low level
- the low-level cost is only terminal latent distance to that subgoal

So the actual decomposition is:

1. high-level planner invents a sequence of latent macro-actions
2. the model rolls those forward in latent space
3. only the first predicted intermediate latent is used as the current subgoal
4. low-level planner tries to reach that subgoal in a short primitive horizon

This is important because the first subgoal is not explicitly optimized for short-horizon reachability by the low-level planner.

## What the current training/eval horizon mismatch looks like

### Training span is short

In `config/train/data/hi_pusht.yaml`:

- training sequence length is `history_size + max_span = 3 + 15 = 18`

In `config/train/hi_lewm.yaml`:

- waypoints are sampled with `max_span=15`
- `num_waypoints=5`
- the model only learns macro transitions contained inside that short window

That means the high level only ever sees local transitions over at most 15 dataset steps.

### d=25 already needs careful planner settings

From `jobs/eval/hi/PLANNING_HPARAM_RESULTS.md`, the main pattern is:

- hierarchical d25 is weak with the default longer low-level horizon
- performance improves a lot when low-level horizon drops from 5 to 2
- best completed d25 result is the hierarchical-soft `low_h=2` setup

That already suggests the hierarchy is fragile and heavily dependent on making the low level solve a very short, easy reachability problem.

### d=50 scales difficulty faster than our hierarchy scales competence

For PushT `d=50`:

- goal is 50 steps ahead
- eval budget is still 50
- the low-level controller in the best d25 setup only executes 5 env steps before replanning and optimizes over 10 primitive steps (`low_horizon=2`, `action_block=5`)

So at `d=50`, the system needs to chain several good intermediate subgoals in a row. That is a much harder requirement than d25, because now errors in subgoal quality accumulate across many replans.

## Main reasons I think d=50 fails

## 1. The high-level model is trained on local transitions, but used as a long-horizon subgoal generator

This is the biggest issue.

The high-level predictor is trained on short waypoint jumps sampled from windows of at most 15 steps. At eval time, it is asked to repeatedly produce useful subgoals over a full `d=50` task.

Even if each subgoal is only locally ahead, d50 requires:

- consistency across many replans
- low drift under repeated autoregressive rollout
- subgoals that stay on a feasible manipulation manifold

Nothing in training explicitly teaches that.

The model only learns one-step waypoint prediction under teacher forcing on offline data. It does not learn:

- compounding-error robustness
- planning-aware recoverability
- what kinds of latent subgoals are easy for the low-level CEM to realize

This kind of mismatch usually gets much worse as horizon grows, which is exactly what you observe from d25 to d50.

## 2. The high-level planner searches over free continuous latent actions, so it can exploit the model

In `hi_policy.py`, the high-level CEM directly optimizes latent macro-actions inside a box calibrated from encoded action chunks.

That means the planner is not restricted to a discrete library of real macro-actions from the dataset. It can synthesize arbitrary latent vectors inside the bounding box.

This is attractive, but it creates a classic model-exploitation problem:

- training sees macro-actions produced by the encoder on real action chunks
- planning can propose many continuous latent codes that are unlikely under the encoder
- the high-level predictor may still assign them optimistic rollouts

At short horizon this can sometimes still work. At d50, chaining several such optimistic latent actions is much more likely to drift off-manifold.

The latent prior calibration is also fairly weak:

- bounds come from percentile boxes
- default calibration uses fixed `chunk_len=5`

But the training chunks vary with waypoint gaps up to 15, and the desired effective subgoal difficulty at d50 is not necessarily well represented by a 5-step bound estimate.

So the search space is only loosely tied to actual reachable macro-actions.

## 3. The first high-level subgoal is not directly constrained to be low-level reachable

This is the second architectural issue I would focus on.

High-level cost is terminal distance to the final goal after a multi-step latent rollout. But the controller only executes the first subgoal from that planned chain.

That creates a disconnect:

- CEM optimizes a full high-level chain
- execution only consumes the first predicted waypoint
- the first waypoint is good only insofar as it helps the final latent goal in model space
- it is not explicitly optimized to be reachable by the low-level controller in 5 to 10 primitive steps

This matters more at d50 because the d50 wrappers increase high-level planning horizon relative to the d25 best setting:

- best d25 family uses `HIGH_HORIZON=1`
- d50 wrappers try `HIGH_HORIZON=2`, `3`, or `4`

With `HIGH_HORIZON>1`, the first subgoal can become a speculative stepping stone that looks good to the high-level model but is awkward for the low level to actually hit.

This is exactly the kind of issue that would make d25 `high_h=1` look okay while d50 degrades when we simply increase planner compute and horizon.

## 4. More CEM compute alone is unlikely to solve the problem

The zip contents and your note both point the same way:

- d50 scripts mainly scale samples / iterations / planner horizon
- there is an overnight sweep dedicated to that
- the sweep still performs badly

That makes sense. If the dominant problem is model mismatch or subgoal feasibility mismatch, then stronger optimization just finds better solutions to the wrong objective.

In fact, more compute can make model exploitation worse by optimizing harder against a misspecified latent cost.

## 5. The hierarchy is currently missing an explicit notion of manipulation progress

PushT is not just locomotion in latent space. The task requires controlled block interaction.

A useful hierarchical subgoal for PushT should probably preserve at least one of these properties:

- move toward a latent state that corresponds to improved block placement
- keep the agent in contact-relevant configurations
- stay inside the manifold where the frozen low-level model remains accurate

But the current high-level objective is only:

- predict next waypoint latent from action chunk latent

and the current high-level planning objective is only:

- minimize final latent distance to goal

There is no explicit signal for:

- contact progress
- block-centric progress
- low-level reachability
- subgoal stability under replanning

This is a likely reason the model works worse as task difficulty increases.

## 6. The best d25 result may already be telling us the hierarchy only works when the low level does most of the real work

The strongest d25 result comes from making the low-level horizon small and easy, not from making the high level more ambitious.

My reading of that is:

- the current hierarchy helps only when it proposes very local targets
- the low level then does most of the control burden
- once the task needs a genuinely good sequence of subgoals, the current high level is not strong enough

That interpretation is consistent with the poor d50 behavior.

## Why HLWM-style d50 compute scaling does not transfer cleanly here

The d50 wrappers borrow the paper-style idea of increasing CEM samples and iterations. That is reasonable as a first attempt, but our model is not the same kind of hierarchy.

HLWM-style hierarchical planning assumes a high-level abstraction that is actually useful for long-horizon composition. Our current model is a LeWM low-level with a learned macro-action layer on top, trained only by local waypoint prediction.

So importing the d50 compute schedule from HLWM does not fix the core issue:

- our high level has not yet earned the right to be trusted as a longer-horizon planner

## What I would try next

Below is the priority order I would use.

## Priority 1: test whether the real bottleneck is high-level horizon, not CEM budget

Run d50 with the most conservative hierarchical policy possible:

- `HIGH_HORIZON=1`
- `HIGH_REPLAN_INTERVAL=3` and `5`
- keep `LOW_HORIZON=2`
- keep low-level `TOPK` high

Reason:

- if `HIGH_HORIZON>1` is causing speculative, low-level-unreachable first subgoals, this should help immediately
- it isolates whether the failure is mainly caused by multi-step high-level planning

I would expect this to outperform the `HIGH_HORIZON=3/4` defaults even if total planning compute is lower.

## Priority 2: constrain high-level search to real macro-actions

Instead of planning over arbitrary continuous latent vectors, try one of:

- nearest-neighbor retrieval from a dataset bank of encoded macro-actions
- CEM over mixture components fitted to encoded macro-actions
- CEM proposals snapped to nearest valid macro-action embeddings

Reason:

- this directly attacks model exploitation
- it keeps high-level planning on the action manifold actually seen in training

If d50 improves under a constrained latent action space, that is strong evidence that free latent search is the problem.

## Priority 3: make the first subgoal explicitly low-level reachable

Three variants are worth trying:

1. high-level cost on the first predicted subgoal only
2. blended cost:
   `goal_distance(final)` + `lambda * reachability_cost(first_subgoal)`
3. simulate one low-level rollout inside high-level scoring for the first subgoal

Reason:

- the currently executed object is the first subgoal, not the final high-level rollout state
- d50 likely needs feasible short-hop subgoals more than clever long latent plans

This is the most directly architecture-aware fix.

## Priority 4: retrain the high level on spans that better match d50 execution

Current training max span is 15. I would try:

- increase `wm.high_level.waypoints.max_span` to 25 or 30
- try fewer waypoints with larger gaps
- try fixed-stride waypoint sampling before random sampling

Reason:

- d50 requires a stable chain of local-but-meaningful subgoals
- fixed stride can reduce training distribution noise and make macro-action semantics more consistent

I would start with something like:

- `num_waypoints=3`
- `strategy=fixed_stride`
- `stride=5` or `10`

This gives the high-level model a much cleaner abstraction than random partitions of a 15-step window.

## Priority 5: train the high level on action-embedding tokens, not raw grouped actions

Right now the latent action encoder consumes raw grouped actions. A cleaner variant is:

- pass primitive actions through the frozen low-level action encoder first
- aggregate those action embeddings into a macro-action latent

Reason:

- it keeps the high-level action abstraction in the same space already used by the low-level predictor
- it may make the macro-actions easier for the high-level predictor to interpret

I do not think this is the first thing to try, but it is a good medium-priority modeling cleanup.

## Priority 6: add a consistency loss for autoregressive high-level rollout

Right now training is teacher-forced local prediction. I would add at least one of:

- rollout loss over 2 to 4 waypoint transitions
- scheduled sampling
- latent subgoal consistency under repeated rollout

Reason:

- d50 failure smells like compounding error
- pure teacher forcing can look good on one-step latent prediction and still plan badly

This is one of the most principled fixes if we want the high level to support true long-horizon planning.

## Priority 7: use a macro-action library or waypoint decoder for diagnostics

Before changing too much, I would add debugging for:

- nearest dataset macro-action to each planned latent action
- norm / percentile position of planned latent actions vs training distribution
- latent distance from current state to chosen subgoal
- low-level success at reaching sampled subgoals offline

Reason:

- this will tell us whether d50 fails because subgoals are off-manifold, too far, or simply inconsistent

Without this, planner tuning is mostly blind.

## Experiments I would not prioritize first

I would not start with:

- larger CEM budgets only
- larger `HIGH_HORIZON` only
- more aggressive continuous latent search

Those are all optimization-side changes. The evidence so far points more toward representation/objective mismatch than insufficient search.

## Concrete next experiment set

If I had to choose only a few runs, I would do these first.

### Set A: conservative d50 planner sanity check

- `HIGH_HORIZON=1`
- `HIGH_REPLAN_INTERVAL=3`
- `LOW_HORIZON=2`
- keep high and low sample counts moderate to high

Goal:

- check whether d50 mainly breaks when we ask the high level to plan multi-step chains

### Set B: constrained macro-action planning

- keep the same planner as Set A
- replace free latent action sampling with nearest-neighbor macro-action candidates from data

Goal:

- detect whether high-level model exploitation is the dominant issue

### Set C: longer-span high-level retraining

Retrain with:

- `max_span=25` or `30`
- fixed stride waypoints
- `num_waypoints=3`
- keep low level frozen

Then reevaluate with conservative d50 planner settings.

Goal:

- align high-level training better with d50 execution demands

## Bottom line

My current best explanation is:

1. the low-level LeWM is fine
2. the high-level module can model short local macro transitions
3. but at d50 we are asking it to be a robust subgoal planner under repeated replanning
4. its training objective does not enforce low-level reachability or long-horizon consistency
5. continuous latent CEM then exploits those weaknesses

So I do not think the main problem is "not enough search".

I think the main problem is:

- the high-level abstraction is not yet aligned with the execution problem we give it on PushT `d=50`

If I had to bet on the highest-value fix, it would be:

- keep the high level very local at eval (`HIGH_HORIZON=1`)
- constrain macro-actions to the data manifold
- retrain the high level with cleaner, longer, more structured waypoint spans

