# Slides: Latent Space Evaluation — Further Investigation

These slides cover the new diagnostic experiments from `latent_eval/`. Add them after the existing diagnostic tracks (Oracle Subgoal, CEM Exploitation, Subgoal Reachability).

All images are in `latent_eval/results/analysis/`.

---

## Slide 1 — Motivation: What Does the Model Actually Do in Latent Space?

**Title:** Beyond Success Rate — Inspecting the Latent Trajectories Directly

**Body:**
- Previous diagnostics confirmed CEM exploits the world model and subgoals are out-of-distribution.
- Open question: *how* does the model's latent trajectory differ from the expert's?
- New evaluation: record full 192-dim latent vectors at every timestep for both expert and model rollout across 50 episodes.

**Key question to answer:**
> Is the failure a *temporal phase* problem (right manifold, wrong timing) or a *manifold* problem (wrong region of latent space entirely)?

> **Image:** `summary.png` — four-panel overview of all five metrics; use as a teaser on this slide.

---

## Slide 2 — Experimental Setup

**Title:** Latent Evaluation Protocol

**Bullet points:**
- Checkpoint: `hope2` (best continuous frozen P2; 84% at d=25)
- 50 PushT episodes, d=25, eval budget = 50 steps
- At each timestep:
  - **Expert latent**: ViT-tiny encoder applied to dataset frame
  - **Model latent**: same encoder applied to the agent's actual observed frame
- Also record **subgoal latent** `z_subgoal` at every high-level CEM replan (every 5 steps)
- 4,884 latent rows × 192 dimensions; 5 offline metrics computed

*(No image needed — diagram or text only)*

---

## Slide 3 — Finding 1: Structural Decoupling, Not Drift

**Title:** Expert and Model Latents Are Orthogonal Throughout Every Episode

**Two-column layout:**

Left — table:

| Metric | Value |
|--------|-------|
| Cosine similarity (mean) | **≈ 0.000** |
| Cosine similarity (range) | −0.220 to +0.430 |
| L2 distance (mean) | **19.80** |
| L2 distance (std) | 1.13 |

Right — key insight:
- Mean cosine ≈ 0 → vectors are **orthogonal** on average — unrelated orientations in latent space.
- Wide cosine range (−0.22 to +0.43) → per-step variation exists, but the mean is structurally zero.
- L2 ≈ 20 throughout — no growth, no shrinkage. Decoupling is present from step 0.

**Takeaway:** Not a compounding prediction error. Structural, episode-wide displacement.

> **Image:** `drift.png` — cosine similarity (top panel) and L2 distance (bottom panel) per timestep, mean ± std over trajectories, dashed lines at replan boundaries.

---

## Slide 4 — Finding 2: Model Moves Slower in Latent Space

**Title:** Model Velocity is 30% Lower Than Expert — And Drops at Replan Steps

**Table:**

| Source | Mean ‖Δz‖ | At replan steps | At other steps |
|--------|:---------:|:---------------:|:--------------:|
| Expert | 1.456     | 1.482           | 1.451          |
| Model  | 1.015     | **0.933**       | 1.034          |

**Interpretation:**
- Model latent representation changes less per step → lower effective task progress in latent space.
- No spike at replan boundaries → CEM solutions are temporally stable / homogeneous across replan events.
- Expert shows no periodic structure — consistent with continuous expert demonstrations.

> **Image (left):** `velocity.png` — mean ‖Δz‖ per timestep for expert (blue) and model (red), dashed replan lines.  
> **Image (right):** `velocity_ratio.png` — model/expert velocity ratio per timestep; dips below 1 at replan boundaries visible.

---

## Slide 5 — Finding 3: Subgoals Don't Target the Right Temporal Horizon

**Title:** Subgoal Temporal Alignment — U-Shaped, Not Peaked at Offset 5

**Bar chart + table (all 2,334 subgoal records):**

| Offset | % of all subgoals | % of replan-only (n=479) |
|--------|:-----------------:|:------------------------:|
| 1      | **30.7%**         | **29.4%**                |
| 2–4    | 20.8%             | 20.6%                    |
| **5 (expected)** | **5.3%** | **4.8%**             |
| 6–8    | 16.5%             | 16.1%                    |
| 9      | **26.9%**         | **29.0%**                |

**Key message:**
- Mean ≈ 4.75 ≈ 5 is **misleading** — it arises from averaging a bimodal (U-shaped) distribution.
- **~30%** of subgoals match best at offset=1: degenerate collapse — subgoal so far off-manifold that the trivially next expert step is the nearest neighbor at any offset.
- **~27%** match at offset=9: subgoal points beyond the replan horizon.
- Only **~5%** correctly target the expected 5-step lookahead — and this holds even for freshly replanned subgoals (replan-only column).

> **Image (left):** `subgoal_best_offset_hist.png` — histogram of best-matching offsets, all records; red dashed line at offset=5.  
> **Image (right):** `subgoal_best_offset_replan_only_hist.png` — same, replan steps only; U-shape persists.  
> **Image (bottom, optional):** `subgoal_offsets.png` — mean L2 distance to expert at each offset; shows which offset is closest on average.

---

## Slide 6 — Finding 4: Fréchet Distance Confirms Systematic Distributional Shift

**Title:** The Entire Latent Distribution is Shifted — Consistently Across All Episodes

**Table:**

| Scope | Fréchet Distance |
|-------|:----------------:|
| Global (all trajectories pooled) | 186.89 |
| Per-episode mean | **385.99** |
| Per-episode std  | 36.45 |
| Per-episode min / max | 243.12 / 456.40 |

**Reading the numbers:**
- Global FD < per-episode mean: pooling widens both distributions, increasing overlap.
- Low std relative to mean → shift is **not caused by outlier episodes**; it is uniform across all 50.
- Expert and model latent clouds occupy fundamentally different regions of 192-dim space.

> **Image (left):** `frechet.png` — per-episode Fréchet distance bar chart; red dashed line at global FD.  
> **Image (right):** `frechet_hist.png` — distribution of per-episode FD values; narrow histogram confirms uniformity.

---

## Slide 7 — Finding 5: DTW = L2 → Manifold Problem, Not Timing Problem

**Title:** Temporal Re-Alignment Provides Zero Benefit

**Key diagnostic:**

| Metric | Mean |
|--------|:----:|
| DTW distance | 105.48 |
| Aligned L2   | 103.42 |
| DTW / L2 ratio | **1.025** (std=0.067, min=1.00, max=1.34) |

**Why this matters:**
- If the model were on the *correct* manifold but *phase-shifted*, DTW would warp the time axis → DTW ≪ L2.
- DTW ≥ L2 for virtually every episode (ratio median = 1.0) → no temporal warping helps.
- **Conclusion: manifold problem.** The model visits a different region of latent space — not a time-shifted version of the expert's region.

> **Image (left):** `dtw.png` (left panel) — scatter of DTW vs aligned L2 per episode; points lie on or above the diagonal.  
> **Image (right):** `dtw.png` (right panel) — histogram of DTW/L2 ratio; mass concentrated at 1.0.

---

## Slide 8 — Summary: Five-Metric Diagnosis

**Title:** Latent Evaluation — What We Now Know

**Table:**

| Analysis | Key number | Implication |
|----------|-----------|-------------|
| Drift (cosine + L2) | Cosine mean ≈ 0, L2 ≈ 19.80 flat | Structural decoupling from step 0 |
| Velocity | Model 30% slower; lower at replan steps | Low latent dynamics; CEM solutions homogeneous |
| Subgoal offsets | Only 5.3% at expected offset=5 (U-shaped) | Subgoals not reliably targeting correct horizon |
| Fréchet distance | Per-episode FD = 386 ± 36 | Systematic distributional shift, all episodes |
| DTW vs L2 | Ratio mean = 1.025 | Not a phase problem — a manifold problem |

**Overall diagnosis (one sentence):**
> The model inhabits a different, slower-moving region of latent space throughout every episode, and CEM-generated subgoals cannot reliably bridge back to the expert's manifold.

> **Image:** `summary.png` — four-panel summary (drift cosine, velocity, Fréchet histogram, DTW scatter); use as full-slide figure.

---

## Slide 9 — Connection to Earlier Diagnostics and Fix Direction

**Title:** This Confirms the CEM Exploitation Hypothesis — and Points to the Fix

**Left: Earlier evidence**
- Oracle subgoals → 70% success (low-level capable if given good targets).
- Open CEM error = 0.0108 vs teacher error = 0.1177 at d=25 → CEM finds adversarial off-manifold states.

**Right: New evidence**
- Cosine mean ≈ 0 at every step → adversarial states are categorically off-manifold.
- Subgoal offset=1 mass (30.7%) → most CEM subgoals collapse to trivially nearest expert state.
- DTW / L2 ≈ 1.0 → re-ordering time does not help; the model is in the wrong place.

**Fix direction (already explored):**
- **VQ codebook + snapped CEM**: constrains search to the discrete set of valid latent macro-actions → 48% at d=50 vs. 38% unconstrained.
- **Empirical code-usage priors + entropy controls**: prevents codebook collapse to Index-0.
- **Open question:** do constrained subgoals show a non-degenerate offset distribution (peaked at 5, not U-shaped)?

*(No image needed — connect back to existing results slides)*
