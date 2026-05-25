# Hierarchical Diagnostics README

This directory is the analysis and provenance layer for the hierarchical PushT fail analysis.

The goal of this README is to give a future LLM or human a single canonical context file that answers:

- what diagnostics exist
- why each diagnostic was added
- what each one actually computes in code
- what files launch them
- what outputs they write
- how the diagnostic stack evolved over time

It is not meant to replace the raw reports. It is meant to tell you how to navigate them.

## Recommended Reading Order

If you want the shortest path to the current story, read these in order:

1. [README.md](/gpfs/home2/scur0200/main/roadmap/analysis/README.md)
2. [hi_pusht_executive_report_2026-05-09.md](/gpfs/home2/scur0200/main/roadmap/analysis/hi_pusht_executive_report_2026-05-09.md)
3. [hi_diagnostics_report_2026-05-09.md](/gpfs/home2/scur0200/main/roadmap/analysis/hi_diagnostics_report_2026-05-09.md)
4. [hi_acting_diagnostics_plan_2026-05-09.md](/gpfs/home2/scur0200/main/roadmap/analysis/hi_acting_diagnostics_plan_2026-05-09.md)
5. [paper_diagnostics_context.md](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/paper_diagnostics_context.md)

If you want the narrow hope2-only story, also read:

- [hope2_experiment_2_teacher_forced_vs_open_loop_high_level_error.md](/gpfs/home2/scur0200/main/roadmap/analysis/hope2_experiment_2_teacher_forced_vs_open_loop_high_level_error.md)
- [hope2_experiment_4_generated_subgoal_reachability.md](/gpfs/home2/scur0200/main/roadmap/analysis/hope2_experiment_4_generated_subgoal_reachability.md)

## High-Level History

### Before diagnostics

The project started from a flat LeWorldModel baseline and added a hierarchical controller on top of it.

The early failure pattern was:

- flat planning stayed competitive
- hierarchical planning could help in a narrow regime
- default hierarchical planning often underperformed, especially when the horizon was short or the online loop was unstable

This broader context is summarized in [hi_pusht_executive_report_2026-05-09.md](/gpfs/home2/scur0200/main/roadmap/analysis/hi_pusht_executive_report_2026-05-09.md).

### Phase 1: Offline diagnostics added

Commit:

- `a24d2a8` on 2026-05-09, message: `implemented and executed diagnostics tests`

Files introduced in that step:

- [scripts/diagnostics/hi_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_diagnostics.py)
- [scripts/diagnostics/run_hi_diagnostic.py](/gpfs/home2/scur0200/main/scripts/diagnostics/run_hi_diagnostic.py)
- [jobs/eval/hi/diagnostics/run_diagnostics_matrix.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/run_diagnostics_matrix.sh)
- [jobs/eval/hi/diagnostics/submit_diagnostics_matrix.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/submit_diagnostics_matrix.sh)
- [jobs/eval/hi/diagnostics/checkpoints_diagnostics.txt](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/checkpoints_diagnostics.txt)

This phase added four fully offline diagnostics:

- `macro_manifold`
- `teacher_vs_open_loop`
- `dataset_subgoal_reachability`
- `generated_subgoal_reachability`

The original full writeup is [hi_diagnostics_report_2026-05-09.md](/gpfs/home2/scur0200/main/roadmap/analysis/hi_diagnostics_report_2026-05-09.md).

### Phase 2: Acting diagnostics added

Commit:

- `6a0250b` on 2026-05-10, message: `diagnostics acting`

Files introduced in that step:

- [scripts/diagnostics/hi_acting_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_acting_diagnostics.py)
- [scripts/diagnostics/run_hi_acting_diagnostic.py](/gpfs/home2/scur0200/main/scripts/diagnostics/run_hi_acting_diagnostic.py)
- [jobs/eval/hi/acting_diagnostics/run_acting_diagnostics_matrix.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/acting_diagnostics/run_acting_diagnostics_matrix.sh)
- [jobs/eval/hi/acting_diagnostics/submit_acting_diagnostics_matrix.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/acting_diagnostics/submit_acting_diagnostics_matrix.sh)
- [jobs/eval/hi/acting_diagnostics/checkpoints_acting.txt](/gpfs/home2/scur0200/main/jobs/eval/hi/acting_diagnostics/checkpoints_acting.txt)

This phase added four online-environment diagnostics:

- `oracle_subgoal_acting`
- `low_level_reality_gap`
- `generated_subgoal_acting`
- `online_hierarchical_logging`

The original planning rationale is [hi_acting_diagnostics_plan_2026-05-09.md](/gpfs/home2/scur0200/main/roadmap/analysis/hi_acting_diagnostics_plan_2026-05-09.md).

The actual results were later appended to [hi_diagnostics_report_2026-05-09.md](/gpfs/home2/scur0200/main/roadmap/analysis/hi_diagnostics_report_2026-05-09.md) as the acting-diagnostics addendum.

### Phase 3: Paper packaging and story figures

Commit:

- `c99d912` on 2026-05-22, message: `Finalize diagnostics and decoder probe workflows`

Files introduced in that step:

- [jobs/eval/hi/paper_diagnostics/submit_hope2_paper_diagnostics.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/submit_hope2_paper_diagnostics.sh)
- [jobs/eval/hi/paper_diagnostics/run_render_hope2_paper_diagnostics.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/run_render_hope2_paper_diagnostics.sh)
- [scripts/diagnostics/render_hi_paper_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/render_hi_paper_diagnostics.py)

The current `paper_diagnostics` directory later accumulated extra helpers for decoder/story figures:

- [jobs/eval/hi/paper_diagnostics/submit_hope2_decoder_story_jobs.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/submit_hope2_decoder_story_jobs.sh)
- [jobs/eval/hi/paper_diagnostics/run_compute_hope2_decoder_story_artifacts.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/run_compute_hope2_decoder_story_artifacts.sh)
- [jobs/eval/hi/paper_diagnostics/run_render_hope2_decoder_story_figures.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/run_render_hope2_decoder_story_figures.sh)
- [jobs/eval/hi/paper_diagnostics/run_render_hope2_story_figures.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/run_render_hope2_story_figures.sh)
- [scripts/diagnostics/render_hi_decoder_diagnostic_stories.py](/gpfs/home2/scur0200/main/scripts/diagnostics/render_hi_decoder_diagnostic_stories.py)
- [scripts/diagnostics/render_hi_story_figures.py](/gpfs/home2/scur0200/main/scripts/diagnostics/render_hi_story_figures.py)

Important distinction:

- these scripts do not define new scientific diagnostics
- they package existing diagnostics into paper tables and decoder-based qualitative figure bundles

## Current Core Claim

The diagnostics stack was built to test a specific failure hypothesis:

- the main benchmark should be read as a failure of the current hierarchical controller to dominate the baseline
- not as proof that hierarchical planning is inherently wrong

The diagnostics narrow the useful regime:

- hierarchy helps more when the horizon is long
- splitting a long plan into stages is often better than one long macro step
- at shorter horizons, the high-level planner can become an extra source of error
- the remaining gap often comes from a combination of high-level prediction drift, generated subgoal quality, low-level online reachability, and online replanning instability

## Shared Terminology

The reports and code use a consistent shorthand.

### Distances

- `d25` means `goal_offset_steps = 25`
- `d50` means `goal_offset_steps = 50`

### High-level horizon

- `hh1` means one high-level stage
- `hh2` means two high-level stages

The code converts `goal_offset_steps` into high-level tokens using:

- `goal_tokens = ceil(goal_offset_steps / frame_skip)`

With `frame_skip = 5`, this becomes:

- `d25 -> 5` high-level tokens
- `d50 -> 10` high-level tokens

The token partition is computed by `partition_total` in [scripts/diagnostics/hi_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_diagnostics.py).

Examples:

- `d25, hh1 -> [5]`
- `d25, hh2 -> [2, 3]`
- `d50, hh1 -> [10]`
- `d50, hh2 -> [5, 5]`

### Low-level horizon

- `lh2`, `lh3`, `lh5` mean the low-level CEM plans over 2, 3, or 5 low-level action tokens

### Low-level receding horizon

- `lrh1` means execute one low-level token block before replanning
- `lrh5` means execute five low-level token blocks before replanning

### Group factor

The code infers a `group` factor from the model:

- `group = macro_input_dim / raw_action_dim`

That factor determines how many raw action rows are bundled into one macro action token.

This matters because some diagnostics work in:

- high-level macro-action token space
- low-level action token space
- raw environment step space

## Shared Runtime Assumptions

These assumptions are common across the diagnostics unless a specific experiment overrides them.

### Data and model loading

Offline diagnostics use:

- dataset: `pusht_expert_train`
- model loader: `stable_worldmodel.policy.AutoCostModel`
- cache resolution from `STABLEWM_DIAGNOSTICS_CACHE_DIR`, `STABLEWM_CACHE_DIR`, `STABLEWM_HOME`, or a local fallback

Acting diagnostics add:

- eval config: [config/eval/hi_pusht.yaml](/gpfs/home2/scur0200/main/config/eval/hi_pusht.yaml)
- real environment: `stable_worldmodel.World`

### Action normalization

All diagnostics normalize raw actions with a `StandardScaler` fit on the full non-NaN dataset action column.

This happens in:

- `build_action_scaler` in [scripts/diagnostics/hi_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_diagnostics.py)

### Valid-start sampling

The diagnostics sample only contiguous within-episode windows.

The helper:

- `contiguous_valid_starts`

checks:

- same episode id across the required chunk
- monotone `step_idx` increments when available

### Offline versus acting sample count

Offline diagnostics default to:

- `num_eval_samples = 256`
- `num_empirical_chunks = 4096`
- `reference_latent_pool_size = 4096`

Acting diagnostics default to:

- `num_eval = 50`
- `num_reference_samples = 4096`

### CEM budgets

The matrix launchers use:

- for `d25` offline rows:
  - high level: `900` samples, `20` iters, `topk=10`
  - low level: `300` samples, `30` iters, `topk=150`
- for `d50` offline rows:
  - high level: `1500` samples, `40` iters, `topk=10`
  - low level: `900` samples, `20` iters, `topk=150`

Acting phase-1 production rows use:

- high level: `1500` samples, `40` iters, `topk=10`
- low level: `900` samples, `20` iters, `topk=150`

## Diagnostic Families

There are eight actual diagnostics, split into an offline family and an acting family.

## Offline Family

Entry points:

- [scripts/diagnostics/run_hi_diagnostic.py](/gpfs/home2/scur0200/main/scripts/diagnostics/run_hi_diagnostic.py)
- [scripts/diagnostics/hi_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_diagnostics.py)
- [jobs/eval/hi/diagnostics/run_diagnostics_matrix.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/run_diagnostics_matrix.sh)

Summary TSV outputs:

- [jobs/eval/hi/diagnostics/logs/summary_macro_manifold.tsv](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/logs/summary_macro_manifold.tsv)
- [jobs/eval/hi/diagnostics/logs/summary_teacher_vs_open_loop.tsv](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/logs/summary_teacher_vs_open_loop.tsv)
- [jobs/eval/hi/diagnostics/logs/summary_dataset_subgoal_reachability.tsv](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/logs/summary_dataset_subgoal_reachability.tsv)
- [jobs/eval/hi/diagnostics/logs/summary_generated_subgoal_reachability.tsv](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/logs/summary_generated_subgoal_reachability.tsv)

### 1. `macro_manifold`

Code path:

- `run_macro_action_manifold_diagnostic` in [scripts/diagnostics/hi_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_diagnostics.py)

Question it answers:

- are the macro-action latents selected by the high-level CEM planner drifting away from the empirical macro-action distribution seen in the dataset?

What it does exactly:

- samples contiguous raw action chunks from the dataset for the required span length
- converts those chunks into normalized macro-action tokens
- encodes them with `model.encode_macro_actions`
- builds a reference distribution for each span length
- runs high-level CEM toward the final goal latent
- compares the CEM-selected macro latents against the empirical reference

What it logs:

- dataset mean norm
- selected mean norm
- selected best norm
- elite cloud norm
- Mahalanobis distance percentiles relative to the empirical macro reference

Why it exists:

- there is a train-test mismatch between:
  - dataset-derived macro action chunks used in training
  - CEM-generated macro latents used at inference

What a bad result means:

- the planner is choosing macro latents that are far from the data manifold
- one long macro step may be a structurally bad abstraction

Important caveat:

- this diagnostic is fully offline and only probes the macro latent space
- it says nothing about online execution by itself

### 2. `teacher_vs_open_loop`

Code path:

- `run_high_level_teacher_vs_open_loop_diagnostic` in [scripts/diagnostics/hi_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_diagnostics.py)

Question it answers:

- is the high-level predictor itself unstable under rollout, or are CEM-selected macro sequences the main problem?

What it does exactly:

- samples valid starts and builds:
  - `z_init`
  - `macro_seq` from true dataset action chunks
  - `target_seq` from encoded future images at macro boundaries
- computes three rollout modes:
  - teacher-forced rollout using true latents between steps
  - open-loop rollout under the true macro sequence
  - open-loop rollout under a CEM-selected macro sequence

What it logs:

- `teacher_forced_mse_mean`
- `open_loop_true_mse_mean`
- `open_loop_cem_mse_mean`
- per-step teacher/open-true/open-CEM errors
- `open_loop_true_over_teacher`
- `open_loop_cem_over_open_true`
- `high_predictor_instability_flag`
- `planner_prior_issue_flag`

Important implementation detail:

- `high_predictor_instability_flag` is set when `open_loop_true_over_teacher > 1.5`
- `planner_prior_issue_flag` is set when `open_loop_cem_over_open_true > 1.5`

Why it exists:

- `hh2` performance can fail either because:
  - the high-level model drifts once it feeds on its own predictions
  - the selected macro sequence is itself model-incompatible

What a bad result means:

- large `open_loop_true / teacher` means predictor compounding error
- large `open_loop_cem / open_true` means planned macros are worse than true macros under the model

Important caveat:

- this is still offline and latent-space only
- it does not prove that an online controller will succeed

### 3. `dataset_subgoal_reachability`

Code path:

- `run_dataset_subgoal_reachability_diagnostic` in [scripts/diagnostics/hi_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_diagnostics.py)

Question it answers:

- can the low-level planner reach real future latents from the same trajectory?

What it does exactly:

- samples valid starts
- chooses true future latents from the same trajectory at offsets `2`, `3`, and `5`
- for each offset, runs low-level CEM from the current latent to that true future latent
- measures the terminal latent error of the best low-level rollout endpoint

What it logs:

- per-offset terminal latent error for `offset2`, `offset3`, `offset5`
- overall mean terminal error across those offsets
- selected mean/best action norms in the full JSON

Why it exists:

- before blaming generated subgoals, we need to know whether the low level can hit sensible targets at all

What a bad result means:

- if this fails on real dataset futures, generated subgoals are not the first problem

Important caveat:

- the default offsets are low-level token offsets, not arbitrary env-step offsets
- the true future target row is indexed as `start + offset * group_factor`

### 4. `generated_subgoal_reachability`

Code path:

- `run_generated_subgoal_reachability_diagnostic` in [scripts/diagnostics/hi_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_diagnostics.py)

Question it answers:

- are the high-level generated subgoals both reachable by the low level and close to realistic future latents?

What it does exactly:

- high-level CEM first optimizes a macro latent sequence toward the final goal latent
- the model rolls out that selected macro sequence to generate stage subgoals
- for each generated subgoal:
  - low-level CEM optimizes toward the generated latent
  - terminal latent error is measured
  - nearest dataset-latent distance is measured against a sampled global latent pool
  - nearest same-trajectory-future distance is measured against future latents reachable from that start

What it logs:

- per-stage low-level terminal cost
- per-stage terminal latent error
- per-stage nearest dataset distance
- per-stage nearest same-trajectory distance
- generated subgoal norm and Mahalanobis stats in the full JSON

Why it exists:

- it tests whether generated waypoints are plausible control targets, not just whether the high-level can minimize its own latent objective

What a bad result means:

- high terminal error means the low level cannot realize the generated target
- large same-trajectory distance means the generated target may be temporally or semantically misaligned with the actual future

## Acting Family

Entry points:

- [scripts/diagnostics/run_hi_acting_diagnostic.py](/gpfs/home2/scur0200/main/scripts/diagnostics/run_hi_acting_diagnostic.py)
- [scripts/diagnostics/hi_acting_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_acting_diagnostics.py)
- [jobs/eval/hi/acting_diagnostics/run_acting_diagnostics_matrix.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/acting_diagnostics/run_acting_diagnostics_matrix.sh)

Summary TSV outputs:

- [jobs/eval/hi/acting_diagnostics/logs/summary_oracle_subgoal_acting.tsv](/gpfs/home2/scur0200/main/jobs/eval/hi/acting_diagnostics/logs/summary_oracle_subgoal_acting.tsv)
- [jobs/eval/hi/acting_diagnostics/logs/summary_low_level_reality_gap.tsv](/gpfs/home2/scur0200/main/jobs/eval/hi/acting_diagnostics/logs/summary_low_level_reality_gap.tsv)
- [jobs/eval/hi/acting_diagnostics/logs/summary_generated_subgoal_acting.tsv](/gpfs/home2/scur0200/main/jobs/eval/hi/acting_diagnostics/logs/summary_generated_subgoal_acting.tsv)
- [jobs/eval/hi/acting_diagnostics/logs/summary_online_hierarchical_logging.tsv](/gpfs/home2/scur0200/main/jobs/eval/hi/acting_diagnostics/logs/summary_online_hierarchical_logging.tsv)

### 5. `oracle_subgoal_acting`

Code path:

- `run_oracle_subgoal_acting` in [scripts/diagnostics/hi_acting_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_acting_diagnostics.py)

Question it answers:

- if we replace the high-level generator with true oracle stage targets, can the low-level controller execute them in the real environment?

What it does exactly:

- prepares real eval episodes by resetting the world from dataset-conditioned starts
- partitions the total horizon into stages
- constructs oracle stage targets from the true future latents at the stage boundaries
- runs a staged hierarchical policy in the real environment with those oracle stage targets
- re-encodes the actual env state at stage ends and at the final step

What it logs:

- `stage1_terminal_latent_error_mean`
- `final_terminal_latent_error_mean`
- `goal_progress_mean`
- `success_rate`
- average per-block model error, actual error, and `reality_gap_mean`
- detailed `stage_end_events` and `low_block_events` in JSON

Why it exists:

- this is the cleanest test of whether the online low-level executor is fundamentally capable

What a bad result means:

- if oracle subgoals fail badly, the high-level generator is not the only problem
- the online low-level planner or the model-to-env mismatch is failing too

### 6. `low_level_reality_gap`

Code path:

- `run_low_level_reality_gap` in [scripts/diagnostics/hi_acting_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_acting_diagnostics.py)

Question it answers:

- does the low-level planner exploit model error, looking good in rollout but failing in the real environment?

What it does exactly:

- picks oracle subgoals at offsets `2`, `3`, and `5`
- converts each offset into env-step duration via `offset * frame_skip`
- plans toward that target in the model
- executes the chosen low-level blocks in the real environment
- re-encodes the actual final env state

What it logs:

- `model_error_mean`
- `actual_error_mean`
- `reality_gap_mean = actual_error - model_error`
- `goal_progress_mean`
- `success_rate`
- aggregate `overall_actual_error_mean`
- aggregate `overall_reality_gap_mean`

Why it exists:

- the offline reachability diagnostic can say the model can plan to a target
- this diagnostic checks whether that success survives real execution

What a bad result means:

- low model error plus high actual error means the low-level CEM is exploiting rollout mismatch

### 7. `generated_subgoal_acting`

Code path:

- `run_generated_subgoal_acting` in [scripts/diagnostics/hi_acting_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_acting_diagnostics.py)

Question it answers:

- are generated stage subgoals valid intermediate waypoints when the controller tries to execute them online?

What it does exactly:

- runs the real staged hierarchical controller, not oracle targets
- lets the high level generate stage targets
- executes low-level planning toward those generated subgoals online
- logs the actual stage-end env latents
- computes temporal validity by comparing each generated subgoal to the nearest same-trajectory future latent

What it logs:

- `step1_stage_end_actual_error_mean`
- `step2_stage_end_actual_error_mean`
- `step1_offset_error_token_mean`
- `step2_offset_error_token_mean`
- `final_terminal_latent_error_mean`
- `success_rate`
- `high_plan_events`, `low_block_events`, and `offset_info` in JSON

Important implementation detail:

- expected stage offsets are cumulative high-level token counts
- actual temporal validity is computed from the nearest future-latent match in the trajectory
- offset error is `nearest_future_offset_token_mean - expected_offset_token`

Why it exists:

- offline generated-subgoal reachability only says the low level can hit a generated latent in the model
- this acting version checks whether the generated midpoint is a useful online waypoint at the right time

What a bad result means:

- large negative stage-1 offset error means the first generated waypoint is temporally early
- good latent reachability plus bad offset error means the waypoint can be hit but is still the wrong waypoint

### 8. `online_hierarchical_logging`

Code path:

- `run_online_hierarchical_logging` in [scripts/diagnostics/hi_acting_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_acting_diagnostics.py)

Question it answers:

- where does the full online hierarchical loop destabilize during real execution?

What it does exactly:

- runs the standard hierarchical policy online in the environment
- logs high-level replans, low-level execution blocks, and per-step latent distances
- records subgoal churn across replans

What it logs:

- `success_rate`
- `final_terminal_latent_error_mean`
- `mean_distance_current_to_subgoal`
- `mean_distance_current_to_final_goal`
- `mean_subgoal_churn_mse`
- `mean_reality_gap`
- detailed `high_plan_events`
- detailed `low_block_events`
- detailed `step_events`

Why it exists:

- this is the closest diagnostic to the real benchmark failure mode
- it is designed to expose bad first subgoals, subgoal churn, or weak real progress despite low model cost

What a bad result means:

- large churn means replanning is rewriting the objective too aggressively
- low progress and low predicted cost together suggest model exploitation
- poor success with acceptable oracle acting points back to high-level online instability

## One Planned Diagnostic That Became Embedded Instead

The acting-diagnostics plan proposed a standalone low-level action OOD experiment.

That did not become a separate top-level diagnostic name in the current code.

Instead, action OOD summaries were embedded inside:

- `low_block_events` in staged acting diagnostics
- `high_plan_events` and `low_block_events` in online logging

The helper functions are:

- `action_token_summary`
- `macro_action_summary`

in [scripts/diagnostics/hi_acting_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_acting_diagnostics.py).

So if you are looking for a ninth diagnostic, it does not exist as a standalone matrix row.

## Matrix Definitions

### Offline matrix

Defined in [jobs/eval/hi/diagnostics/run_diagnostics_matrix.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/run_diagnostics_matrix.sh).

Rows `1-20` are:

- `1-4`: `macro_manifold` over `d25/d50 x hh1/hh2`
- `5-8`: `teacher_vs_open_loop` over `d25/d50 x hh1/hh2`
- `9-16`: `dataset_subgoal_reachability` over `d25/d50 x lh1/lh2/lh3/lh5`
- `17-20`: `generated_subgoal_reachability` over `d25/d50 x hh1/hh2`, with `lh2`

### Acting matrix

Defined in [jobs/eval/hi/acting_diagnostics/run_acting_diagnostics_matrix.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/acting_diagnostics/run_acting_diagnostics_matrix.sh).

Rows `1-12` are:

- `1-3`: `oracle_subgoal_acting` over `lh2/lh3/lh5`
- `4-6`: `low_level_reality_gap` over `lh2/lh3/lh5`
- `7-8`: `generated_subgoal_acting` for `hh1/lh2/lrh1` and `hh2/lh2/lrh1`
- `9-12`: `online_hierarchical_logging` for:
  - `hh1/lh2/lrh1`
  - `hh2/lh2/lrh1`
  - `hh2/lh5/lrh1`
  - `hh2/lh5/lrh5`

## Paper Diagnostics Layer

The current paper-facing directory is:

- [jobs/eval/hi/paper_diagnostics](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics)

This layer is hope2-specific and mostly uses:

- `run_name = hi_lewm_p2_train_hope2_22253175`
- `epoch = 15`

It does three things:

### 1. Re-render paper summary figures and tables

Files:

- [submit_hope2_paper_diagnostics.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/submit_hope2_paper_diagnostics.sh)
- [run_render_hope2_paper_diagnostics.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/run_render_hope2_paper_diagnostics.sh)
- [scripts/diagnostics/render_hi_paper_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/render_hi_paper_diagnostics.py)

These scripts read:

- `summary_teacher_vs_open_loop.tsv`
- `summary_oracle_subgoal_acting.tsv`
- `summary_low_level_reality_gap.tsv`
- `summary_generated_subgoal_acting.tsv`
- `summary_online_hierarchical_logging.tsv`

and produce:

- `teacher_vs_open_loop`
- `failure_ladder_d50`
- `low_level_horizon_sweep`
- `generated_subgoal_offset_error`
- `online_churn_and_success`

### 2. Compute decoder-story artifacts

Files:

- [run_compute_hope2_decoder_story_artifacts.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/run_compute_hope2_decoder_story_artifacts.sh)

This reruns:

- `generated_subgoal_acting`
- `online_hierarchical_logging`

for the hope2 paper setting and saves rich `.json` and `.npz` artifacts for figure rendering.

### 3. Render qualitative story figures

Files:

- [run_render_hope2_decoder_story_figures.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/run_render_hope2_decoder_story_figures.sh)
- [scripts/diagnostics/render_hi_decoder_diagnostic_stories.py](/gpfs/home2/scur0200/main/scripts/diagnostics/render_hi_decoder_diagnostic_stories.py)
- [run_render_hope2_story_figures.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/run_render_hope2_story_figures.sh)
- [scripts/diagnostics/render_hi_story_figures.py](/gpfs/home2/scur0200/main/scripts/diagnostics/render_hi_story_figures.py)

These are decoder-based visualization layers that turn saved latents into image panels.

They are useful because they let you see:

- teacher-forced versus open-loop forecast divergence
- oracle versus generated subgoal acting outcomes
- temporal validity failures of generated subgoals
- online replanning churn over multiple high-level plans

Canonical context for the current paper-diagnostics outputs is:

- [paper_diagnostics_context.md](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/paper_diagnostics_context.md)

## Figure Map For Writing

If you are assembling the paper figures, keep the quantitative summary layer and the decoder-visualization layer separate.

Recommended main figures:

- `failure_ladder_d50`
- `generated_subgoal_offset_error`

Optional qualitative main figure if space permits:

- `oracle_vs_generated.png`

Recommended appendix figures:

- `temporal_validity.png`
- `online_replanning.png`
- `probe_sanity.png`
- `offline_forecast_stage_1.png`
- `offline_forecast_stage_2.png`

### What each figure is actually showing

- `failure_ladder_d50` is a quantitative bar chart built from summary TSV rows plus the best D50 flat baseline extracted from `baseline_matrix_results_2026-05-21.md`.
- `generated_subgoal_offset_error` is a quantitative bar chart over stagewise token-offset error for generated subgoals.
- `oracle_vs_generated.png` is a decoder-rendered qualitative acting comparison. It is the closest visual companion to the oracle and generated rows inside `failure_ladder_d50`.
- `temporal_validity.png` is a decoder-rendered qualitative visualization of generated-subgoal timing mismatch. It is the closest visual companion to `generated_subgoal_offset_error`.
- `online_replanning.png` is a decoder-rendered qualitative visualization of the online hierarchical row of the failure ladder, specifically replanning churn and mismatch between planned and reached states.
- `probe_sanity.png` checks whether the latent-to-pixel decoder is visually trustworthy enough to use for the other qualitative figures.
- `offline_forecast_stage_1.png` and `offline_forecast_stage_2.png` visualize the teacher-vs-open-loop forecast diagnostic stage by stage.

### Can the two main figures be replaced by decoder images?

`generated_subgoal_offset_error`: effectively yes.

- The decoder-backed version is `temporal_validity.png`.
- It does not aggregate into a bar chart; it shows concrete failure cases with `Context`, `Target`, `Generated`, `Closest Match`, `Reached`, and `Final Goal`.
- If you want a decoder qualitative figure in the slot of the summary plot, this is the correct counterpart.

`failure_ladder_d50`: not exactly, at least not with the current artifact stack.

- The current failure ladder includes `Baseline best`, and the decoder-story workflow does not have saved flat-baseline rollout artifacts.
- The existing decoder figures only cover the hierarchical rows:
  - `oracle_vs_generated.png` covers oracle versus generated staged acting.
  - `online_replanning.png` covers the online hierarchical failure mode.
- So there is no faithful one-file decoder replacement for the full `failure_ladder_d50` chart today.

Practical recommendation:

- keep `failure_ladder_d50` as a quantitative main figure
- keep `generated_subgoal_offset_error` as a quantitative main figure if you want summary statistics
- use `oracle_vs_generated.png` as the qualitative companion for the ladder
- use `temporal_validity.png` as the qualitative companion for generated-subgoal offset error

## Current Interpretation Of The Stack

The current diagnostic story is:

- the low level is not fundamentally incapable
- real future subgoals are reachable offline
- oracle staged subgoals can work online, especially with `lh2`
- one long macro step is usually the worst high-level regime
- split horizons often improve subgoal geometry and latent reachability
- generated subgoals can still be temporally wrong even when they are easy to hit in latent space
- the full online loop can still fail because replanning churn and stage-to-stage instability remain

In other words:

- the diagnostics do not support “hierarchy is impossible”
- they support “the current hierarchical controller is unstable and narrow-regime”

## Which Files Are Canonical Versus Derived

Canonical code:

- [scripts/diagnostics/hi_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_diagnostics.py)
- [scripts/diagnostics/hi_acting_diagnostics.py](/gpfs/home2/scur0200/main/scripts/diagnostics/hi_acting_diagnostics.py)
- [jobs/eval/hi/diagnostics/run_diagnostics_matrix.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/run_diagnostics_matrix.sh)
- [jobs/eval/hi/acting_diagnostics/run_acting_diagnostics_matrix.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/acting_diagnostics/run_acting_diagnostics_matrix.sh)

Canonical raw reports:

- [hi_diagnostics_report_2026-05-09.md](/gpfs/home2/scur0200/main/roadmap/analysis/hi_diagnostics_report_2026-05-09.md)
- [paper_diagnostics_context.md](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/paper_diagnostics_context.md)

Derived/narrow notes:

- [hope2_experiment_2_teacher_forced_vs_open_loop_high_level_error.md](/gpfs/home2/scur0200/main/roadmap/analysis/hope2_experiment_2_teacher_forced_vs_open_loop_high_level_error.md)
- [hope2_experiment_4_generated_subgoal_reachability.md](/gpfs/home2/scur0200/main/roadmap/analysis/hope2_experiment_4_generated_subgoal_reachability.md)
- [d25_failure_root_cause_report.md](/gpfs/home2/scur0200/main/roadmap/analysis/d25_failure_root_cause_report.md)
- [d25_hier_soft_run_analysis_and_next_steps.md](/gpfs/home2/scur0200/main/roadmap/analysis/d25_hier_soft_run_analysis_and_next_steps.md)

Planning note:

- [hi_acting_diagnostics_plan_2026-05-09.md](/gpfs/home2/scur0200/main/roadmap/analysis/hi_acting_diagnostics_plan_2026-05-09.md)

Miscellaneous:

- [status_.md](/gpfs/home2/scur0200/main/roadmap/analysis/status_.md)

## If You Need To Reconstruct The Story Quickly

Use this template:

1. Start from the benchmark claim in [hi_pusht_executive_report_2026-05-09.md](/gpfs/home2/scur0200/main/roadmap/analysis/hi_pusht_executive_report_2026-05-09.md).
2. Use offline diagnostics to separate:
   - manifold drift
   - predictor instability
   - low-level latent reachability
   - generated-subgoal plausibility
3. Use acting diagnostics to separate:
   - oracle low-level capability
   - model-to-env reality gap
   - temporal validity of generated subgoals
   - online replanning churn
4. Use `paper_diagnostics` only as the packaging layer for the current hope2 paper story.

## Bottom Line

The diagnostic stack was built in three steps:

- first, measure offline high-level and low-level failure modes
- second, move the same questions into the real environment
- third, package the current hope2 results into paper figures and decoder-based stories

The stack is coherent. The important point is not just which numbers are good or bad. The important point is that each diagnostic removes one ambiguity:

- `macro_manifold` removes “maybe CEM stays on-manifold”
- `teacher_vs_open_loop` removes “maybe the high level is stable under rollout”
- `dataset_subgoal_reachability` removes “maybe the low level cannot hit sensible targets”
- `generated_subgoal_reachability` removes “maybe generated subgoals are both plausible and reachable”
- `oracle_subgoal_acting` removes “maybe the online low level cannot execute even correct midpoints”
- `low_level_reality_gap` removes “maybe model rollout and env execution match closely”
- `generated_subgoal_acting` removes “maybe generated midpoints are temporally valid online”
- `online_hierarchical_logging` removes “maybe the closed-loop controller is stable once all pieces are combined”

That is the real history of the analysis.
