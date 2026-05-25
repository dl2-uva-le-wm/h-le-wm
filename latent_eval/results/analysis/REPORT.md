# Latent Evaluation Analysis Report

**Dataset:** 4,884 rows · 192-dim latents · 50 trajectories  
**Replan interval:** 5 steps  
**PCA for DTW / per-episode Fréchet:** 20 components (59.2% variance)  
**Checkpoint:** `hi_lewm_p2_train_hope2_22253175` (best continuous frozen P2; 84% at d=25)  
**Expert latents:** ViT-tiny encoder on dataset frames. **Model latents:** same encoder on live rollout frames.

---

## Extension 1: Per-step Drift

**Metric:** cosine similarity and L2 distance between `expert[t]` and `model[t]` at each timestep.

| Statistic | Cosine similarity | L2 distance |
|-----------|:-----------------:|:-----------:|
| Mean      | ≈ 0.000           | 19.80       |
| Std       | 0.083             | 1.13        |
| Min       | −0.220            | 13.58       |
| Max       | +0.430            | 22.43       |

**Key finding:** The mean cosine similarity is essentially zero across all timesteps, meaning expert and model latent vectors are on average **orthogonal** — occupying entirely different orientations in latent space. The non-trivial standard deviation (0.083) and range (−0.22 to +0.43) show per-step variation: individual timesteps occasionally produce moderate alignment, but the average is structurally near zero. L2 distance stays ≈ 20 throughout episodes (std = 1.13) — the separation does not grow or shrink with time.

**Interpretation:** The model is not gradually diverging from the expert trajectory — it is decoupled from the start. This is not a compounding prediction error: it is a structural, episode-wide displacement in latent space.

---

## Extension 2: Latent Velocity

**Metric:** step-to-step displacement `‖z[t+1] − z[t]‖` per source.

| Source | Mean velocity | Std  | Max  |
|--------|:-------------:|:----:|:----:|
| Expert | 1.456         | 0.851| 6.15 |
| Model  | 1.015         | 0.770| 7.82 |

**Model is ~30% slower than expert** in latent space on average.

**Replan-boundary check (t mod 5 == 0):**

| Step type     | Model velocity | Expert velocity |
|---------------|:--------------:|:---------------:|
| Replan steps  | 0.933          | 1.482           |
| Other steps   | 1.034          | 1.451           |

Model velocity is lower at replan boundaries than at intermediate steps — the opposite of what CEM-induced churn would produce. If CEM replanning produced diverse new subgoals, there would be spikes at `t mod 5 == 0`. The absence of spikes indicates CEM solutions are stable across replan events (low plan diversity), or the latent representation changes smoothly despite replanning. Expert velocity shows no periodic structure (1.482 vs 1.451), consistent with continuous demonstrations without discrete replan events.

---

## Extension 3: Subgoal Temporal Alignment

**Metric:** for each recorded subgoal `z_sub` at timestep `t`, find the temporal offset `k ∈ {1..9}` that minimises `‖z_sub − expert[t+k]‖`.

### All subgoal records (n = 2,334)

| Statistic | Best offset |
|-----------|:-----------:|
| Mean      | 4.75        |
| Std       | 3.32        |
| Median    | 4.0         |

| Offset | Count | % of total |
|--------|------:|----------:|
| 1      | 716   | **30.7%** |
| 2      | 196   | 8.4%      |
| 3      | 153   | 6.6%      |
| 4      | 135   | 5.8%      |
| **5 (expected)** | **123** | **5.3%** |
| 6      | 117   | 5.0%      |
| 7      | 121   | 5.2%      |
| 8      | 146   | 6.3%      |
| 9      | 627   | **26.9%** |

### Replan-only records (n = 479, freshly computed subgoals)

| Offset | Count | % of replan |
|--------|------:|:-----------:|
| 1      | 141   | **29.4%**   |
| 2      | 39    | 8.1%        |
| 3      | 27    | 5.6%        |
| 4      | 33    | 6.9%        |
| **5 (expected)** | **23** | **4.8%** |
| 6      | 20    | 4.2%        |
| 7      | 24    | 5.0%        |
| 8      | 33    | 6.9%        |
| 9      | 139   | **29.0%**   |

**Key finding:** Both all-records and replan-only distributions are strongly **U-shaped**, not peaked at the expected offset of 5. The mean of ≈ 4.75–4.91 ≈ 5 is misleading — it arises from averaging a bimodal distribution, not from concentration at the expected value.

**Interpretation:**
- **Offset=1 mass (≈30%):** Many subgoals collapse to the trivially nearest expert state. The subgoal is so far off the expert manifold that the closest expert state at any offset is the immediate next step — a degenerate result.
- **Offset=9 mass (≈27%):** A separate mode where subgoals point beyond the replan horizon, suggesting the high-level model overshoots in temporal lookahead.
- The correct temporal framing (offset=5) is reliably achieved in fewer than 5% of replan events.

---

## Extension 4: Fréchet Distance

**Metric:** FD between Gaussian fits of expert and model latent clouds (distribution-level shift).

| Scope            | FD value |
|------------------|:--------:|
| Global           | 186.89   |
| Per-episode mean | 385.99   |
| Per-episode std  | 36.45    |
| Per-episode min  | 243.12   |
| Per-episode max  | 456.40   |

**Key finding:** FD values are very large. Global FD (186.89) is lower than per-episode mean (385.99) because pooling all trajectories widens both distributions, increasing overlap. Per-episode FD is consistently large — standard deviation (36.45) is small relative to the mean (385.99), confirming the distributional shift is **systematic across all episodes**, not an artefact of outlier trajectories.

**Interpretation:** Expert and model latent clouds occupy substantially different regions of the 192-dim space. The model has not learned to reproduce the expert's latent distribution, not even approximately.

---

## Extension 5: DTW vs Aligned L2

**Metric:** DTW distance on 20-dim PCA projections (59.2% variance), compared against aligned L2.

| Metric       | Mean   | Std   | Min   | Max    |
|--------------|:------:|:-----:|:-----:|:------:|
| DTW          | 105.48 | —     | —     | —      |
| Aligned L2   | 103.42 | —     | —     | —      |
| DTW/L2 ratio | 1.025  | 0.067 | 1.000 | 1.338  |

**Key finding:** DTW/L2 ratio has median = 1.0 and mean = 1.025. DTW distance ≥ aligned L2 in virtually all episodes. Temporal re-alignment provides **no benefit**.

For a model suffering purely from a temporal offset (correct manifold, wrong phase), DTW ≪ L2 — warping the time axis would collapse the distance. The absence of this effect means the deviation is **not a temporal offset problem** — the model is in the wrong region of latent space regardless of time-axis alignment.

---

## Summary and Diagnosis

| Extension | Finding | Implication |
|-----------|---------|-------------|
| 1: Drift | Cosine mean ≈ 0 (std=0.083), L2 ≈ 19.80 (std=1.13), flat across time | Structural latent decoupling — not transient, not compounding |
| 2: Velocity | Model 30% slower; velocity lower at replan steps | Low latent dynamics; CEM plans are temporally stable |
| 3: Subgoals | U-shaped offset distribution; only 5.3% at expected offset=5 | Subgoal framing unreliable; CEM subgoals predominantly off-manifold |
| 4: Fréchet | Global FD=186.89, per-episode mean=386 ± 36 | Large, systematic distributional shift — all episodes |
| 5: DTW | DTW ≈ L2 (ratio mean=1.025) | Problem is manifold shift, not temporal offset |

**Overall diagnosis:**  
The model's latent trajectories are not on the expert's latent manifold. This is not a timing/phase issue (DTW provides no gain), not a growing divergence (L2 variance is small over time), and not confined to outlier episodes (Fréchet std is small relative to mean). The model inhabits a different, slower-moving region of latent space throughout every episode.

The subgoal analysis directly links to the behavioural failure: CEM-generated subgoals cluster at offset=1 (degenerate, off-manifold collapse) or offset=9 (beyond the replan horizon), with only ~5% correctly targeting the 5-step lookahead. This is consistent with the CEM exploitation hypothesis established by the open-loop error analysis (Open CEM errors of 0.0108 vs teacher errors of 0.1177 at d=25) — the optimizer finds adversarial states that appear optimal to the model but are unreachable in reality.

**Possible root causes:**
1. **Encoder not shared or fine-tuned:** if the model's observation encoder differs from the one used to record expert latents, representations are incomparable by construction.
2. **Observation distribution shift:** the agent's rollout observations differ from expert demonstrations (visual differences, goal-conditioning artefacts), producing different encoder outputs even with the same weights.
3. **Latent collapse / mode averaging:** the model's policy outputs low-variance latents that average over expert modes, reducing both velocity and increasing Fréchet separation.
4. **Goal conditioning mismatch:** subgoals sampled from a different distribution than the encoder's training distribution push the model into off-manifold regions.

**Recommended next steps:**
- Verify the same encoder checkpoint is used for both expert and model latent extraction.
- Visualise 2D UMAP/PCA projections of expert vs model clouds to confirm the manifold gap is visible in low-dimensional projection.
- Check per-step observation images for expert vs model to rule out observation distribution shift.
- Investigate whether the offset=1 subgoal mass correlates with specific trajectory types (success vs failure, early vs late in episode).
- Re-run subgoal offset analysis using constrained CEM (snapped to VQ codebook) to verify whether manifold constraints shift the distribution toward offset=5.
