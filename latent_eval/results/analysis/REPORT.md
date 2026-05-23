# Latent Evaluation Analysis Report

**Dataset:** 4,884 rows · 192-dim latents · 50 trajectories  
**Replan interval:** 5 steps  
**PCA for DTW:** 20 components (59.2% variance)

---

## Extension 1: Per-step Drift

**Metric:** cosine similarity and L2 distance between expert[t] and model[t] at each timestep.

| Statistic | Cosine similarity | L2 distance |
|-----------|:-----------------:|:-----------:|
| Mean      | −0.005            | 19.86       |
| Std       | 0.008             | 0.11        |
| Min       | −0.018            | 19.56       |
| Max       | 0.009             | 20.03       |

**Key finding:** Cosine similarity hovers at ~0 across all timesteps (range −0.018 to +0.009). Near-zero cosine means the model and expert latent vectors are approximately **orthogonal** — they occupy entirely different orientations in latent space. L2 distance is constant at ≈20 with negligible variance, indicating the model does not drift *toward or away* from the expert trajectory over time — it maintains a fixed, large separation throughout the episode.

**Interpretation:** The separation is not a transient effect that grows or shrinks — it is a structural, episode-long decoupling. The model is not visiting the same latent region as the expert at any point in the trajectory.

---

## Extension 2: Latent Velocity

**Metric:** step-to-step displacement `‖z[t+1] − z[t]‖` per source.

| Source | Mean velocity | Std  | Max  |
|--------|:-------------:|:----:|:----:|
| Expert | 1.456         | 0.851| 6.15 |
| Model  | 0.882         | 0.667| 6.77 |

**Model is ~39% slower than expert** in latent space on average.

**Replan-boundary check (t mod 5 == 0):**

| Step type     | Model velocity | Expert velocity |
|---------------|:--------------:|:---------------:|
| Replan steps  | 0.760          | 1.482           |
| Other steps   | 0.910          | 1.450           |

Counterintuitively, model velocity is *lower* at replan boundaries than at intermediate steps. CEM replanning should produce spikes (replan churn) if the new goal differs from the old one — the absence of spikes suggests either the model's latent representation changes smoothly despite replanning, or the CEM solutions are stable across replan events (low plan diversity).

Expert velocity shows no periodic structure (1.482 vs 1.450), consistent with continuous expert demonstrations without discrete replan events.

---

## Extension 3: Subgoal Temporal Alignment

**Metric:** for each recorded subgoal z_sub at timestep t, find the temporal offset k ∈ {1..9} that minimises `‖z_sub − expert[t+k]‖`.

| Statistic | Best offset |
|-----------|:-----------:|
| Mean      | 4.73        |
| Std       | 3.30        |
| Median    | 4.0         |

**Distribution of best-matching offsets:**

| Offset | Count | % of total |
|--------|------:|----------:|
| 1      | 714   | 30.6%     |
| 2      | 188   | 8.1%      |
| 3      | 154   | 6.6%      |
| 4      | 140   | 6.0%      |
| **5 (expected)** | **135** | **5.8%** |
| 6      | 125   | 5.4%      |
| 7      | 124   | 5.3%      |
| 8      | 146   | 6.3%      |
| 9      | 608   | 26.1%     |

**Key finding:** The distribution is strongly U-shaped, not peaked at the expected offset of 5. ~30.6% of subgoals align best at offset=1 (immediate next step) and ~26.1% at offset=9 (maximum checked). Only 5.8% align best at offset=5.

The mean of 4.73 ≈ 5 is misleading — it arises from averaging a bimodal distribution, not from concentration at the expected value.

**Interpretation:**
- The large offset=1 mass suggests many subgoals collapse to the trivially nearest expert state (degenerate, likely because the latent subgoal is far from all expert states and the nearest-neighbor at any offset is the immediate next step).
- The large offset=9 mass suggests a separate mode where subgoals point ahead beyond the replan horizon.
- The correct temporal framing (offset=5) is not reliably achieved.

---

## Extension 4: Fréchet Distance

**Metric:** FD between Gaussian fits of expert and model latent clouds (distribution-level shift).

| Scope       | FD value |
|-------------|:--------:|
| Global      | 202.7    |
| Per-episode mean | 390.9 |
| Per-episode std  | 31.3  |
| Per-episode min  | 274.9 |
| Per-episode max  | 460.4 |

**Key finding:** FD values are very large. The global FD (202.7) is lower than the per-episode mean (390.9) because pooling all trajectories widens both distributions, increasing overlap. Per-episode FD is consistently high and has low spread (std=31.3 on a mean of 390.9), meaning the distributional shift is **systematic across all episodes**, not an artefact of outlier trajectories.

**Interpretation:** Expert and model latent clouds occupy substantially different regions of the 192-dim space. This confirms the drift analysis — the model has not learned to reproduce the expert's latent distribution, not even approximately.

---

## Extension 5: DTW vs Aligned L2

**Metric:** DTW distance on 20-dim PCA projections (59.2% variance), compared against aligned L2.

| Metric       | Mean   | Std   | Min   | Max    |
|--------------|:------:|:-----:|:-----:|:------:|
| DTW          | 108.44 | 10.97 | 78.78 | 131.29 |
| Aligned L2   | 106.09 | 13.19 | 69.71 | 131.29 |
| DTW/L2 ratio | 1.028  | 0.077 | 1.000 | 1.357  |

**Key finding:** The DTW/L2 ratio has median=1.0 and mean=1.028. DTW distance ≥ aligned L2 in virtually all episodes (75th percentile of ratio = 1.0 exactly). DTW being no smaller than L2 means **temporal re-alignment provides no benefit**.

For a model suffering purely from a temporal offset (correct manifold, wrong phase), DTW ≪ L2 — DTW would "warp" the time axis and collapse the distance. The absence of this effect means the model's deviation from expert is **not a temporal offset problem** — it is a manifold problem: the model is in the wrong region of latent space, regardless of time axis alignment.

---

## Summary and Diagnosis

| Extension | Finding | Implication |
|-----------|---------|-------------|
| 1: Drift | Cosine ≈ 0, L2 ≈ 20, constant across time | Structural latent decoupling — not transient |
| 2: Velocity | Model 39% slower; no replan spikes | Low latent dynamics; CEM plans are temporally stable |
| 3: Subgoals | U-shaped offset distribution; mean ≈ 5 by coincidence | Subgoal framing unreliable; not consistently at horizon |
| 4: Fréchet | Global FD=202.7, per-episode ≈391 | Large, systematic distributional shift |
| 5: DTW | DTW ≈ L2 (ratio ≈ 1.03) | Problem is not temporal offset — it is manifold shift |

**Overall diagnosis:**  
The model's latent trajectories are not on the expert's latent manifold. This is not a timing/phase issue (DTW provides no gain) and not a growing divergence (L2 is flat over time). The model inhabits a different, slower-moving region of latent space throughout every episode. Possible causes:

1. **Encoder not shared or fine-tuned:** if the model's observation encoder differs from the one used to record expert latents, representations are incomparable.
2. **Distribution shift in observations:** the model's rollout observations differ from expert demonstrations (different camera angles, randomisation, etc.), producing different encoder outputs.
3. **Latent collapse / mode averaging:** the model's policy outputs average or low-variance latents, reducing both velocity and FD separation.
4. **Goal conditioning mismatch:** subgoals sampled from a different distribution than the encoder's training distribution push the model into off-manifold regions.

**Recommended next steps:**
- Verify the same encoder checkpoint is used for both expert and model latent extraction.
- Visualise 2D UMAP/PCA projections of expert vs model clouds (install `umap-learn` — currently only PCA is available).
- Check per-step observation images for expert vs model to rule out observation distribution shift.
- Investigate whether subgoal offset=1 mass correlates with specific trajectory types (success vs failure, early vs late in episode).
