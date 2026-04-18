Short answer: **for the hybrid design we converged on, this is mostly right**.
For the **original PDF proposal as written**, it is **not exactly right**, because your PDF says _jointly trained from scratch, no frozen components_, while this code is clearly built around an optional **pretrained/frozen low-level LeWM**.

What is already **correct** in your code for the hybrid version:

- **P2 is LeWM-style**, not HWM-token-interleaving style: you pass `emb` plus an **action-conditioning sequence** to `high_predictor`, which is exactly the LeWM philosophy where actions condition the transformer rather than becoming tokens. That matches LeWM much better.
- Your **latent action encoder** is a **2-layer bidirectional tiny transformer with CLS pooling**, which is exactly the kind of module we settled on and is also consistent with the hierarchical paper’s macro-action encoder idea.
- You are training P2 with **teacher forcing on a sequence of waypoint latents and macro-actions**, not on isolated `(z,l)` pairs. That is the right thing to do.
- `rollout_high()` is **autoregressive** and uses a **sliding context window**. That is also right.

So the overall structure is good.

## The 3 things I would fix before trusting it

### 1. Do **not** share `pred_proj` between P1 and P2

Right now `predict_high()` uses:

```python
preds = self.high_predictor(emb, high_cond)
preds = self.pred_proj(...)
```

but `pred_proj` is the **low-level LeWM projection head**. If you freeze low-level, then P2 is forced to map its hidden states into a projection head trained for **P1 statistics**, not P2 statistics.

That is the single thing I dislike most in your code.

**What I would do instead**

- keep `low_pred_proj`
- add a separate **`high_pred_proj`**
- train `high_pred_proj` together with `high_predictor`

So yes:

- `projector` can stay shared with the encoder
- `pred_proj` should be **split**
- otherwise P2 is unnecessarily constrained

### 2. `get_cost_high()` should require `d_high == latent_action_dim`

This part is too permissive:

```python
if d_high % latent_dim != 0:
    ...
high_action_block = d_high // latent_dim
latent_actions = action_candidates.reshape(b, s, h * high_action_block, latent_dim)
```

That can silently change the **effective horizon** during planning.

For the design you want, the clean rule is:

```python
assert d_high == latent_action_dim
```

Then:

```python
latent_actions = action_candidates
```

Much safer.

### 3. Separate **high-level context length** from **number of sampled waypoints**

Right now you do:

```python
high_num_frames = num_waypoints - 1
high_predictor = ARPredictor(num_frames=high_num_frames, ...)
```

This works, but it couples two concepts that should be separate:

- **how many waypoint transitions you sample in training**
- **how much history P2 can use**

I would add:

- `wm.high_level.history_size`
- `wm.high_level.waypoints.num`

and only make them equal by default.

That gives you much cleaner control.

## One thing that is **not wrong**, but I would still think about

Right now the latent action encoder takes **raw effective actions**:

```python
latent_action_encoder(input_dim=effective_act_dim, ...)
```

This is okay. It can work.

But if you want the most LeWM-consistent version, I would consider:

- first pass each low-level effective action through the **frozen LeWM action encoder**
- then feed the resulting **192-d action tokens** to the tiny transformer

So instead of:

- raw action chunk → tiny transformer → macro-action

you would do:

- raw action chunk → frozen LeWM `action_encoder` → 192-d action tokens → tiny transformer → macro-action

That is not mandatory, but it is cleaner because the high-level action encoder then lives in the **same action embedding space already used by P1**.

## Is the conditioning over a whole sequence of latent actions correct?

Yes. That part of your code is conceptually right.

In training:

- `macro_actions` has shape `(B, T_wp, D_l)`
- `project_macro_to_condition_space()` makes it `(B, T_wp, D_z)`
- `high_predictor(emb, high_cond)` receives a **sequence** of state latents and a **sequence** of conditioned actions

That is exactly the LeWM-style adaptation you wanted: **AdaLN-style conditioning over a temporal array**, not a single action only. LeWM’s predictor is a temporal predictor with causal masking and action conditioning per step, so using the same logic for P2 is reasonable.

## So: is it right?

My honest verdict:

- **For the hybrid proposal we arrived at in the conversation:** **yes, mostly**
- **For the original PDF text:** **no, not exactly**
- **Would I run experiments with this code?** yes
- **Would I trust it as-is without changes?** no

## Final verdict

I’d call it **“80% correct, with 3 important fixes”**:

1. add a **separate `high_pred_proj`**
2. simplify `get_cost_high()` so planner dim equals latent-action dim
3. decouple **P2 history size** from **number of waypoints sampled**

If you want the strongest chance that it actually works, I would also start with:

- **frozen low-level**
- **fixed-stride waypoints first**
- **train only P2 + latent action encoder + macro adapter + high_pred_proj**

That is the version I would bet on first.
