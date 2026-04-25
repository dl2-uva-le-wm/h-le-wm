## Core diagnosis

Your current failure is **not primarily a LeWM representation-collapse or low-level dynamics failure**. The evidence points to a **hierarchical interface and planning-distribution failure**, with a secondary issue of **undertrained / under-capacity high-level dynamics**.

The strongest clue is in your CSV: the same high-level checkpoint goes from very poor short-horizon performance to near-acceptable performance just by changing planning configuration. At `d=25`, the default-ish hierarchy is around **14–42%**, but the tuned “soft lower horizon” variants reach **84–88%**. That is too large a swing to explain by encoder quality alone. It means the model contains some usable dynamics, but the planner is usually asking the hierarchy to do the wrong thing.

The target paper reports PushT success of **89% / 78% / 61%** for hierarchical DINO-WM at `d=25 / 50 / 75`, compared with flat DINO-WM at **84% / 55% / 17%**. Your current best results are roughly **88% at d=25** and **46% at d=50**, with no reliable `d=75` success yet. So you have recovered short-horizon behavior only after planner tuning, but hierarchy is still **below your own flat LeWM baseline at d=50**. The target long-horizon abstraction has not emerged yet. 

## What the CSV says

| Setting                                     |   d=25 |  d=50 | Interpretation                                             |
| ------------------------------------------- | -----: | ----: | ---------------------------------------------------------- |
| Original flat LeWM in your CSV              |    96% |   58% | Low-level LeWM is strong.                                  |
| Original LeWM with naïve horizon 10 at d=25 |    12% |     — | Longer flat horizon catastrophically hurts.                |
| Hi-LeWM default-ish hierarchy               | 14–42% | 8–10% | Bad planner/hierarchy interface.                           |
| Hi-LeWM best tuned hierarchy                |    88% |   46% | Short-horizon mostly recoverable; long-horizon not solved. |
| Target HWM hierarchy                        |    89% |   78% | Your d=50 gap is the real issue.                           |

The “naïve horizon 10” result is especially important: even the strong original LeWM collapses to **12%** when the planning horizon is increased without the right abstraction. That confirms the LeWM paper’s point that longer autoregressive latent planning accumulates error and becomes harder for CEM. LeWM’s planning objective is terminal latent goal matching solved by CEM, and the paper explicitly notes the tradeoff between longer horizon, computation, and model bias. 

The other key CSV pattern is that **low-level horizon 2 is the sweet spot**. With `high_horizon=1`, `low_horizon=2`, and high low-level top-k, you get 84–88% at `d=25`. With `low_horizon=5`, the same family of runs often drops to 14–42%. That tells me the low-level planner is overcommitting toward imperfect high-level subgoals when asked to execute a full 5 model-step segment.

## Architecture diagnosis

Your codebase follows the right high-level idea: keep the original LeWM stack, add a high-level predictor, encode action chunks into latent macro-actions, and use the high-level rollout to generate subgoals for the low-level planner. The report says the default model freezes the low-level LeWM path and trains only the high-level path, with a `LatentActionEncoder`, `high_predictor`, `macro_to_condition`, and separate `high_pred_proj`. 

The issue is that your **macro-action search space is probably far too large**. Your config sets:

* `embed_dim = 192`
* `latent_action_dim = 192`
* `high_horizon = 1–4` in evaluation

So CEM is optimizing over:

* 192 dimensions for `H=1`
* 384 dimensions for `H=2`
* 768 dimensions for `H=4`

That is enormous for CEM with 900–1500 samples. The HWM paper’s analysis explicitly warns that too much latent-action capacity can produce high-level subgoals that are valid in the high-level model but hard for the low-level planner to execute; moderate latent-action dimensionality biases the planner toward reachable subgoals. 

This is likely your biggest architectural mismatch. You imported the idea of latent macro-actions, but not the **compression pressure** that makes high-level CEM tractable.

## Training diagnosis

Your actual training config is **P2-only**:

* low-level training disabled
* encoder frozen
* low predictor frozen
* low action encoder frozen
* SIGReg weight zero
* `alpha=0`, `beta=1`
* 10 epochs
* high-level loss only

That matches the report’s description of the current staged training regime: freeze the working low-level model and train the high-level waypoint predictor only. 

This is a reasonable bootstrapping strategy, but it is **not comparable to the target paper’s PushT training**. In HWM PushT, the high-level DINO-WM is scaled up substantially and trained for **500 epochs**, while your high-level predictor is essentially the same size family as the low-level LeWM predictor and trained for **10 epochs**. 

Your log shows training is not obviously broken: trainable parameters are 12.5M, frozen parameters 18.0M, and the validation high-level prediction loss falls from about 0.72 initially to about 0.019 by the end.  But this only proves that **teacher-forced waypoint prediction on your sampled waypoint distribution** is learnable. It does not prove that open-loop high-level CEM produces reachable subgoals.

The waypoint statistics are another red flag. Your logged mean waypoint gap is about **2.37 model steps**, with max around **7.45 model steps**. With frame skip 5, the model is mostly trained on roughly **12 environment-step macro transitions**, with occasional transitions around **35–37 environment steps**. But you evaluate at `d=50` and want `d=75`. Your high-level model is not being trained often enough on macro-transitions at the required temporal scale.

## Loss-function diagnosis

LeWM’s original recipe uses only two losses: next-embedding prediction plus SIGReg, where SIGReg prevents representation collapse by enforcing Gaussian-distributed embeddings.  Since your encoder is frozen, turning off SIGReg for high-level-only training is not automatically wrong. You are not learning the representation distribution; you are learning transitions in an already-regularized latent space.

The missing losses are different:

1. **No autoregressive high-level rollout loss.**
   Your high-level loss is teacher-forced waypoint MSE. Evaluation uses autoregressive high-level rollouts. This train/test mismatch is exactly where high-level models fail.

2. **No reachability loss.**
   The high-level model is rewarded for predicting the next dataset waypoint under encoded dataset action chunks, but CEM later searches arbitrary latent macro-actions. Nothing penalizes subgoals that are high-level-plausible but low-level-unreachable.

3. **No latent-action prior penalty during planning.**
   Your macro-action encoder produces dataset macro latents, but high-level CEM can optimize off-manifold latent actions. The large `latent_action_dim=192` makes this much worse.

4. **MSE instead of L1.**
   The HWM high-level objective is written with an L1 latent prediction loss. This is probably not the main issue, but switching high-level prediction to L1 is a low-cost ablation. 

Do **not** add pixel reconstruction first. LeWM’s own ablation found decoder loss reduced PushT performance from 96% to 86%, likely because reconstruction encourages irrelevant visual detail. 

## Planning diagnosis

The planner is currently the most visible failure mode.

The target HWM paper uses horizon-specific CEM settings for PushT. For example, at `d=50`, the flat planner uses a longer prediction horizon and strong variance smoothing, while the hierarchical planner uses `high_horizon=4`, `high_samples=1500`, `high_iters=40`, and low-level `horizon=5`, `samples=900`, with variance EMA.  They also sweep CEM samples, iterations, and variance smoothing to construct compute-success Pareto curves. 

But in your results, copying paper-like larger horizons does **not** help. At `d=50`, increasing `high_horizon` from 1 to 2–4 generally hurts. That means your high-level model is not yet reliable enough for multi-step high-level rollout. The current best d=50 run still uses `high_horizon=1`, `low_horizon=2`.

This is the central failure:

> The hierarchy is only working as a short-range subgoal perturbation mechanism, not as a long-horizon abstraction mechanism.

The low-level planner aiming at the high-level subgoal rather than the final goal is correct by design, and your codebase implements exactly that loop.  But if the high-level subgoal is off-manifold, too distant, or not low-level reachable, the low-level planner becomes worse than flat planning.

## What I would debug next

### 1. Establish a clean baseline table first

Run these with identical 50 start-goal pairs:

| Method                                       |                                      d=25 |  d=50 |    d=75 |
| -------------------------------------------- | ----------------------------------------: | ----: | ------: |
| Original LeWM flat, original config          |                                already 96 |    58 | missing |
| Original LeWM flat, HWM-style horizon config |                                       run |   run |     run |
| Hi-LeWM flat fallback                        | rerun only after matching baseline config | rerun |   rerun |
| Hi-LeWM hierarchy best current config        |                                        88 |    46 |     run |

The codebase report already warns that flat fallback inside the hierarchical repo is not automatically baseline-equivalent. Do not interpret `hi flat = 62%` as low-level failure until you exactly match the original LeWM eval path. 

### 2. Add oracle-subgoal evaluation

This is the fastest way to isolate the failure.

For `d=50` and `d=75`, use the dataset frame at an intermediate time as the subgoal:

* `d=50`: subgoal at `t+25`
* `d=75`: subgoals at `t+25` and/or `t+50`

Then run only the low-level planner toward that encoded subgoal.

Interpretation:

* If oracle subgoals succeed, the low-level planner is fine and high-level subgoal generation is failing.
* If oracle subgoals fail, the low-level planner/cost/action blocking is still wrong.
* If oracle subgoals work at `d=50` but not `d=75`, you need multi-subgoal chaining, not just one high-level subgoal.

### 3. Test dataset macro-actions versus CEM macro-actions

Evaluate high-level prediction in two modes:

1. Encode the true action chunk with `latent_action_encoder`, roll high-level once, compare to the true future latent.
2. Let CEM optimize latent macro-actions, roll high-level once, then measure whether the resulting macro-action lies inside the empirical dataset macro-action distribution.

If true macro-actions predict well but CEM macro-actions fail, your issue is **off-manifold latent-action optimization**, not training loss.

### 4. Sweep latent-action dimension aggressively

This is my highest-priority architecture ablation.

Train variants with:

```text
latent_action_dim ∈ {4, 8, 16, 32, 64, 192}
```

Keep `macro_to_condition` as an MLP when dimensions differ.

Expected outcome: 192 will not be best. I would expect 8–32 to be much more CEM-friendly. The HWM paper’s latent-action-dimension analysis supports exactly this tradeoff: enough capacity for valid plans, but not so much that the high-level planner proposes unreachable subgoals. 

### 5. Change waypoint sampling to match d=50/d=75

Your current `random_sorted`, `num=5`, `max_span=15` setup produces mostly short waypoint gaps. Train separate variants:

```text
A: fixed_stride gap = 2 model steps   # ~10 env steps
B: fixed_stride gap = 5 model steps   # ~25 env steps
C: fixed_stride gap = 10 model steps  # ~50 env steps
D: endpoint-biased random gaps in [5, 15]
```

Then evaluate which gap distribution helps `d=50` and `d=75`.

Right now, your high-level model is not clearly trained on the temporal abstraction it is expected to perform.

### 6. Train the high-level model much longer

Your 10-epoch high-level run is useful for debugging, but it is not a fair attempt to match HWM. Run:

```text
50 epochs
100 epochs
300 epochs
```

Do this only after reducing latent-action dimension, otherwise you may simply overfit a CEM-hostile 192D macro space.

### 7. Add rollout loss to high-level training

Start with:

```text
L_high = L_TF + λ_roll L_roll
λ_roll ∈ {0.1, 0.5, 1.0}
```

Use L1 for both terms as an ablation. The point is not that the paper requires this; the point is that your failure appears exactly when high-level predictions are composed. A rollout loss directly targets that failure.

### 8. Constrain high-level CEM to the empirical latent-action prior

For each latent-action dimension setting, estimate empirical macro-action mean/covariance or per-dimension percentiles from the dataset. Then run CEM in whitened coordinates:

```text
l = μ + L ε
ε ~ N(0, I)
clip ε or penalize Mahalanobis distance
```

Track the Mahalanobis distance of the best CEM macro-action. If successful plans have lower prior distance and failed plans have extreme prior distance, you have confirmed off-manifold planning.

## My current causal ranking

1. **Wrong low-level execution horizon for hierarchical subgoals** — already strongly supported by CSV. `low_horizon=2` is much better than 5.
2. **Macro-action latent space too high-dimensional** — 192D makes high-level CEM under-sampled and encourages unreachable subgoals.
3. **Waypoint training distribution too short for d=50/d=75** — mean gap is much smaller than the intended abstraction horizon.
4. **High-level model undertrained relative to target HWM** — 10 epochs versus 500 in the PushT HWM setup.
5. **High-level model under-capacity relative to target HWM** — your high predictor is not scaled like the target high-level model.
6. **Teacher-forcing-only objective does not guarantee open-loop high-level planning quality.**
7. **Flat fallback / metric parsing issues are contaminating interpretation** — especially the block-only metrics, which appear numerically invalid in the CSV.

## What not to chase first

I would not first tune SIGReg, unfreeze the encoder, add reconstruction, or simply increase CEM samples. LeWM’s original PushT result is already 96%, and the paper shows SIGReg is robust across a wide λ range and that reconstruction loss hurts PushT control. 

Your immediate path should be:

1. prove low-level subgoal reachability with oracle subgoals;
2. reduce latent-action dimension;
3. train high-level on longer, controlled waypoint gaps;
4. add high-level rollout diagnostics/loss;
5. only then scale compute and compare to `d=50 / d=75` targets.
 