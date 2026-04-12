# DL2 Detailed Implementation Proposal

## 1) Objective

Implement a **Top-Down Temporal Hierarchy** extension of LeWorldModel (LeWM) that:

1. Preserves a **single shared latent space** and SIGReg-once principle.
2. Adds hierarchical temporal predictors at horizons `1`, `k1`, `k2`.
3. Uses **Inverse Dynamics (ID)** at Levels 2 and 3 (no CEM there).
4. Keeps **short-horizon CEM** only at Level 1, anchored by Level 2 output.
5. Improves long-horizon planning reliability while remaining computationally efficient.

This document translates `DL2_research_porposal.pdf`, `lewm_hierarchical_proposal.pdf`, and `plan.md` into an execution-ready engineering plan for this repository.

---

## 2) Current Repository Baseline (Important)

The current codebase is rooted at:

- `repo root`

Relevant baseline files:

- `module.py`
- `jepa.py`
- `train.py`
- `eval.py`
- `config/train/lewm.yaml`
- `config/train/data/*.yaml`
- `config/eval/*.yaml`

Implementation should target these files/paths directly (not a separate `hi-le-wm/` folder unless explicitly created later).

---

## 3) Architecture To Implement

## 3.1 New/Extended Modules (`module.py`)

Add:

1. `InverseDynamicsModel`
   - Input: concatenated pair `(z_t, z_target)` with shape `(B, 2D)`.
   - Output: macro-action vector `a_tilde` with shape `(B, A_macro)`.
   - MLP-style architecture with LayerNorm + GELU.

2. `ConditionedSingleStepPredictor`
   - Used as `pred3` and `pred2`.
   - Inputs:
     - `z_current` `(B, D)`
     - `macro_action` `(B, A_macro)`
     - optional `z_anchor` `(B, D)` (required at L2)
   - Output:
     - one latent prediction `(B, D)`.
   - Internally reuse existing `Transformer` + `ConditionalBlock`.

3. `ARPredictorAnchored` (extension of `ARPredictor`)
   - Backward-compatible with existing behavior when `z_anchor=None`.
   - Adds midpoint-anchor conditioning to AdaLN signal.
   - Level 1 remains autoregressive over physical actions.

## 3.2 Hierarchical World Model (`hi_jepa.py`)

Create new model class `HiJEPA` with:

- Shared base pieces:
  - encoder
  - action_encoder
  - projector
  - pred_proj
- Hierarchical components:
  - `pred1`: `ARPredictorAnchored`
  - `pred2`: `ConditionedSingleStepPredictor`
  - `pred3`: `ConditionedSingleStepPredictor`
  - `id2`: `InverseDynamicsModel`
  - `id3`: `InverseDynamicsModel`

Key methods:

1. `encode(info)` (compatible with existing JEPA behavior)
2. `predict(...)` for Level 1 AR prediction
3. `train_forward` support path (called externally by `hi_train.py`)
4. `rollout(...)` for candidate action trajectories with Level 1 anchored dynamics
5. `criterion(...)`:
   - Compare terminal rollout embedding to **midpoint anchor** (not distant goal) during hierarchical planning.
6. `get_cost(info_dict, action_candidates)`:
   - Phase 0: encode start/goal
   - Phase 1: L3 ID + prediction
   - Phase 2: L2 ID + prediction
   - Phase 3: Level 1 short-horizon CEM rollout to midpoint anchor

## 3.3 Training Entry Point (`hi_train.py`)

Create new training script parallel to `train.py`:

1. `hi_lejepa_forward(self, batch, stage, cfg)`:
   - computes `L1`, `L2`, `L3`, `SIGReg`, and macro-action regularization.
   - total loss:
     - `loss = L1 + L2 + L3 + lambda_sigreg * L_reg + alpha_act * L_act`.

2. Model construction:
   - instantiate `id2`, `id3`, `pred2`, `pred3`, `pred1`.
   - instantiate `HiJEPA`.

3. Logging:
   - log per-level losses with clear names:
     - `pred_loss_l1`, `pred_loss_l2`, `pred_loss_l3`
     - `sigreg_loss`, `act_reg_loss`, `loss`.

## 3.4 Configs

Add:

1. `config/train/hi_lewm.yaml`
2. `config/eval/hi_pusht.yaml`
3. Equivalent `hi_*.yaml` for `tworoom`, `cube/reacher`, humanoid target env as ready.

Update data configs for hierarchical sequence length:

- `num_steps = history_size + k2_frames`

Important convention:

- Keep `k1_env`, `k2_env` (env steps) explicit in config.
- Derive dataset-frame offsets via `frameskip`:
  - `k1_frames = k1_env // frameskip`
  - `k2_frames = k2_env // frameskip`
- Validate integer divisibility at startup and fail fast otherwise.

---

## 4) Detailed Implementation Work Packages

## WP1: Core Module Development

Files:

- `module.py`

Tasks:

1. Implement `InverseDynamicsModel`.
2. Implement `ConditionedSingleStepPredictor`.
3. Implement `ARPredictorAnchored`.
4. Add shape assertions and docstrings for all new forwards.
5. Ensure no regression for existing LeWM (`train.py`, `jepa.py`) code path.

Acceptance criteria:

- Unit shape checks pass for all new modules.
- Existing `ARPredictor` behavior unchanged when anchors are not used.

## WP2: Hierarchical Model Integration

Files:

- `hi_jepa.py` (new)

Tasks:

1. Build `HiJEPA` class with JEPA-compatible structure.
2. Implement hierarchical `get_cost` with 4 planning phases.
3. Keep tensor movement/device handling robust.
4. Ensure `WorldModelPolicy`/solver can call `get_cost` unchanged.

Acceptance criteria:

- `get_cost` accepts `(B,S,H,action_dim)` candidates.
- Returns `(B,S)` cost tensor without NaNs in smoke tests.

## WP3: Training Pipeline

Files:

- `hi_train.py` (new)
- `config/train/hi_lewm.yaml` (new)
- `config/train/data/*.yaml` (small updates or `hi_` variants)

Tasks:

1. Implement hierarchical forward pass with all loss terms.
2. Add robust index logic for `z_t`, `z_k1`, `z_k2`.
3. Validate `history_size + k2_frames <= num_steps`.
4. Add optional curriculum flag for L2 anchor teacher forcing:
   - modes: `gt_only`, `mix`, `pred_only`.
5. Add wandb metric groups for per-level diagnostics.

Acceptance criteria:

- 1-epoch smoke training runs on PushT.
- losses finite and decreasing trend in first steps.

## WP4: Evaluation/Inference Pipeline

Files:

- `config/eval/hi_*.yaml` (new)
- Existing `eval.py` reused unless compatibility gap is found.

Tasks:

1. Configure short-horizon CEM:
   - `plan_config.horizon = k1_frames`
   - `action_block` and `receding_horizon` tuned per env.
2. Ensure policy model points to hierarchical checkpoint.
3. Add explicit eval scripts/commands for:
   - flat LeWM baseline
   - Hi-LeWM
4. Save run outputs as parseable text + CSV/JSON aggregate.

Acceptance criteria:

- Evaluation loop completes for PushT baseline and Hi-LeWM with identical seed sets.
- Metrics and timing logged in comparable format.

## WP5: Probing + Ablations

Files:

- `roadmap/tasks/` (new scripts/checklists)
- Optional python scripts under `` (if needed)

Tasks:

1. Probes:
   - latent geometry monotonicity vs temporal distance
   - ID fidelity (`pred` from inferred macro-action vs GT target)
   - hierarchy consistency ratio (`rho ~ k1/k2`)
   - CEM convergence speed vs flat baseline.
2. Ablations:
   - no Level 3
   - no Level 2
   - CEM at all levels
   - multi-level SIGReg
   - `k1/k2` sweep.

Acceptance criteria:

- Every ablation has:
  - exact config delta
  - fixed seed list
  - output artifact path
  - summary table entry.

---

## 5) Timeline (6 Weeks, Specific Deliverables)

## Week 1: Foundation + Design Lock

Deliverables:

1. Technical spec freeze:
   - tensor shapes
   - `k1/k2` convention
   - file-level interfaces
2. Environment readiness check (PushT + TwoRoom + Manipulator + Humanoid data decision).
3. Baseline reproducibility:
   - run existing `train.py`/`eval.py` on PushT and store reference metrics.

Exit criteria:

- “Ready-to-implement” checklist approved by all members.

## Week 2: Module + Model Implementation

Deliverables:

1. `module.py` new classes merged.
2. `hi_jepa.py` initial model with working forward/get_cost.
3. Basic unit tests for shapes and forward passes.

Exit criteria:

- smoke script can instantiate `HiJEPA` and run one dummy batch forward.

## Week 3: Training Integration + First Runs

Deliverables:

1. `hi_train.py` integrated with config.
2. `config/train/hi_lewm.yaml` and data config updates.
3. First PushT training runs (short epochs).
4. Loss dashboard with L1/L2/L3/act-reg/sigreg tracked.

Exit criteria:

- stable (non-diverging) first runs with checkpoints saved.

## Week 4: Inference/Planner Integration

Deliverables:

1. Hierarchical eval configs (`config/eval/hi_*.yaml`).
2. End-to-end hierarchical planning on PushT.
3. Timing benchmark script (flat vs hierarchical).

Exit criteria:

- hierarchical evaluation completes and returns success-rate/time metrics.

## Week 5: Ablations + Hyperparameter Sweeps

Deliverables:

1. Full ablation suite execution.
2. `k1/k2` sensitivity experiments.
3. Anchor curriculum experiments if teacher-forcing gap appears.

Exit criteria:

- ablation table complete with reproducible run IDs and configs.

## Week 6: Consolidation + Report

Deliverables:

1. Final benchmark plots/tables.
2. Failure-case analysis.
3. Reproducibility package:
   - commands
   - config list
   - seed list
   - artifact locations.
4. Final report and presentation material.

Exit criteria:

- all claims backed by tracked experiments and artifacts.

---

## 6) Team Responsibilities (Aligned with Proposal Roles)

## 6.1 Niccolò Caselli (Planning/Inference Lead)

Primary ownership:

1. Hierarchical planning pipeline implementation in `hi_jepa.py`.
2. `get_cost` multi-phase logic and policy integration.
3. CEM horizon configuration and timing profiling.
4. Final integration sanity across train/eval checkpoints.

Concrete outputs:

- Working hierarchical `get_cost`.
- Evaluation command matrix (flat vs hierarchical).
- Planner profiling report (time/step, eval runtime).

## 6.2 Francesco Massafra (Planning/Inference Co-Lead)

Primary ownership:

1. Level 1 anchored rollout logic.
2. Solver compatibility checks with `stable_worldmodel`.
3. MPC behavior validation (`receding_horizon`, `action_block` tuning).
4. Failure-case debugging for planning divergence.

Concrete outputs:

- Planner validation scripts.
- CEM convergence comparisons.
- Config recommendations per environment.

## 6.3 Ippokratis Pantelidis (Model/Training Lead)

Primary ownership:

1. New core classes in `module.py`.
2. Hierarchical model assembly and training graph correctness.
3. Loss design implementation and stability checks.
4. Teacher-forcing/mixed-anchor curriculum implementation.

Concrete outputs:

- Stable `hi_train.py` core path.
- Loss/gradient diagnostics.
- Training stability notes and mitigation decisions.

## 6.4 Samuele Punzo (Model/Training Co-Lead, Hyperparameter Owner)

Primary ownership:

1. Config engineering (`config/train/hi_lewm.yaml`, data updates).
2. Hyperparameter sweeps (`lambda`, `alpha`, `macro_action_dim`, `k1/k2`).
3. Experiment orchestration and run bookkeeping.
4. Regression checks vs base LeWM.

Concrete outputs:

- Sweep config set.
- Aggregated metrics tables.
- Recommended default hyperparameters.

## 6.5 Salvatore Lo Sardo (Evaluation/Environment Lead)

Primary ownership:

1. Environment compatibility and dataset readiness (PushT, TwoRoom, Manipulator, Humanoid).
2. Evaluation harness and benchmark protocol standardization.
3. Ablation/probing execution coordination.
4. Result analysis, plots, and final narrative consistency.

Concrete outputs:

- Environment setup README and runbook.
- Benchmark dataset split protocol.
- Final performance and ablation figures.

---

## 7) Cross-Team Coordination Rules

1. Branching:
   - feature branches per WP (`feat/wp1-modules`, etc.).
2. PR policy:
   - minimum 1 reviewer from another role group.
3. Weekly sync:
   - fixed 2 checkpoints/week:
     - technical blockers
     - metric review.
4. Experiment tracking:
   - every run logs:
     - commit hash
     - config path
     - seed
     - dataset version
     - output artifacts.

---

## 8) Validation and “Definition of Done”

Project is complete only if all are satisfied:

1. Code:
   - hierarchical modules/classes merged and documented.
2. Training:
   - stable training on at least PushT and one additional environment.
3. Inference:
   - hierarchical planner runs end-to-end in existing eval pipeline.
4. Evidence:
   - baseline vs hierarchical comparisons with controlled seeds.
5. Probing/ablation:
   - required probe suite and ablation table completed.
6. Reproducibility:
   - runbook enables rerun from clean checkout.

---

## 9) Main Risks and Mitigations

1. Train/inference mismatch at Level 2 (teacher forcing gap)
   - Mitigation: anchor curriculum (`gt -> mixed -> pred`) + scheduled ratio.

2. `k1/k2` unit mismatch (env steps vs dataset frames)
   - Mitigation: explicit derived config fields and startup assertions.

3. ID models learn shortcut/degenerate macro-actions
   - Mitigation: action L2 regularization + ID fidelity probes.

4. Integration incompatibility with existing policy/solver APIs
   - Mitigation: keep `get_cost` signature identical; add smoke tests before long runs.

5. Compute overrun
   - Mitigation: strict staged gating (smoke -> pilot -> full run), stop criteria for unstable configs.

---

## 10) Immediate Next Actions (This Week)

1. Freeze shape and config spec (`k1/k2` conventions).
2. Implement WP1 (`module.py`) and WP2 skeleton (`hi_jepa.py`) first.
3. Add smoke tests for:
   - forward pass
   - `get_cost` output shape/finite values.
4. Start first short PushT training run via `hi_train.py` once smoke passes.

