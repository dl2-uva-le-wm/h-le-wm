# `jobs/eval/hi/paper_diagnostics` Context

## Scope

This note captures the current state of [`jobs/eval/hi/paper_diagnostics`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics), including:

- repo-local inputs, launchers, logs, and smoketest outputs
- scratch report/artifact directories produced by those launchers
- exact table values currently available
- exact image outputs and where to download them
- execution-status notes from the logs

The checkpoint file fixes the analysis to:

```text
RUN_NAME=hi_lewm_p2_train_hope2_22253175
CHECKPOINT_EPOCH=15
POLICY=runs/hi_lewm_p2_train_hope2_22253175/hi_lewm_p2_train_hope2_22253175_epoch_15
```

Source: [`checkpoints_hope2.txt`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/checkpoints_hope2.txt)

## Repo-Local Inventory

### Top-level files

- [`checkpoints_hope2.txt`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/checkpoints_hope2.txt): single checkpoint row used by the paper diagnostics.
- [`submit_hope2_paper_diagnostics.sh`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/submit_hope2_paper_diagnostics.sh): submits the offline array, acting array, then a render job dependent on both.
- [`run_render_hope2_paper_diagnostics.sh`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/run_render_hope2_paper_diagnostics.sh): renders the paper report tables/figures from the offline and acting log roots.
- [`submit_hope2_decoder_story_jobs.sh`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/submit_hope2_decoder_story_jobs.sh): submits decoder-story artifact compute, then dependent story rendering.
- [`run_compute_hope2_decoder_story_artifacts.sh`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/run_compute_hope2_decoder_story_artifacts.sh): computes generated-subgoal and online-hierarchical `.json/.npz` artifacts on CPU.
- [`run_render_hope2_decoder_story_figures.sh`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/run_render_hope2_decoder_story_figures.sh): renders decoder-story figure bundles from the probe run plus artifact `.npz`s.
- [`run_render_hope2_story_figures.sh`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/run_render_hope2_story_figures.sh): renders a broader story-figure bundle from teacher, oracle, generated, online, and paper-table inputs.

### Repo-local directories

- [`logs/offline`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/logs/offline): 20 offline diagnostic array-job `out/err` pairs plus 4 summary TSVs.
- [`logs/acting`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/logs/acting): 12 acting diagnostic array-job `out/err` pairs plus 4 summary TSVs.
- [`output`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/output): render-job logs, decoder-story compute/render logs, a local smoketest figure bundle, and Matplotlib/fontconfig caches.

### Repo-local output files

#### Render and compute job logs

- [`output/run_render_hope2_paper_diagnostics_23048474.out`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/output/run_render_hope2_paper_diagnostics_23048474.out)
- [`output/run_render_hope2_paper_diagnostics_23048474.err`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/output/run_render_hope2_paper_diagnostics_23048474.err)
- [`output/run_compute_hope2_decoder_story_artifacts_23062979.out`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/output/run_compute_hope2_decoder_story_artifacts_23062979.out)
- [`output/run_compute_hope2_decoder_story_artifacts_23062979.err`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/output/run_compute_hope2_decoder_story_artifacts_23062979.err)
- [`output/run_render_hope2_decoder_story_figures_23062980.out`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/output/run_render_hope2_decoder_story_figures_23062980.out)
- [`output/run_render_hope2_decoder_story_figures_23062980.err`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/output/run_render_hope2_decoder_story_figures_23062980.err)

#### Repo-local smoketest images

- [`output/decoder_story_smoketest/teacher_vs_open_loop/teacher_story_00.png`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/output/decoder_story_smoketest/teacher_vs_open_loop/teacher_story_00.png)
- [`output/decoder_story_smoketest/teacher_vs_open_loop/teacher_story_01.png`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/output/decoder_story_smoketest/teacher_vs_open_loop/teacher_story_01.png)
- [`output/decoder_story_smoketest/generated_subgoal_acting/generated_story_00.png`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/output/decoder_story_smoketest/generated_subgoal_acting/generated_story_00.png)
- [`output/decoder_story_smoketest/generated_subgoal_acting/generated_story_01.png`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/output/decoder_story_smoketest/generated_subgoal_acting/generated_story_01.png)
- [`output/decoder_story_smoketest/manifest.json`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/output/decoder_story_smoketest/manifest.json)

The repo-local smoketest manifest records:

```json
{
  "teacher_cases": [
    {"figure": "teacher_story_00.png", "sample_index": 241, "teacher_err_mean": 0.020495355129241943, "open_true_err_mean": 0.02346692979335785, "open_cem_err_mean": 1.2071919441223145},
    {"figure": "teacher_story_01.png", "sample_index": 184, "teacher_err_mean": 0.04167458787560463, "open_true_err_mean": 0.023397430777549744, "open_cem_err_mean": 0.798614501953125}
  ],
  "generated_cases": [
    {"figure": "generated_story_00.png", "sample_index": 19, "success": false, "stage1_offset_error_steps": -24, "final_error_mean": 2.3146069049835205},
    {"figure": "generated_story_01.png", "sample_index": 20, "success": false, "stage1_offset_error_steps": -24, "final_error_mean": 2.256420612335205}
  ],
  "online_cases": [],
  "online_trace_available": false
}
```

## Job Wiring And Realized Output Directories

### Paper diagnostics submission

`submit_hope2_paper_diagnostics.sh` submits:

- offline array: `--array=1-20`
- acting array: `--array=1-12`
- render job: dependency `afterok:${OFFLINE_JOB_ID}:${ACTING_JOB_ID}` and optionally `:${DECODER_DEPENDENCY_JOB_ID}`

The realized paper render log says:

```text
Report directory: /scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474
```

### Decoder story submission

`submit_hope2_decoder_story_jobs.sh` submits:

- compute job, then
- render job dependent on compute completion, with:
  - `GENERATED_ARTIFACT=/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979/generated_subgoal_acting_d50_hh2_lh2_lrh1.npz`
  - `ONLINE_ARTIFACT=/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979/online_hierarchical_logging_d50_hh2_lh2_lrh1.npz`

The realized decoder-story logs say:

```text
Decoder story artifacts: /scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979
Decoder story figures: /scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980
```

### Story-figure bundle

`run_render_hope2_story_figures.sh` defaults to:

- `PAPER_TABLES_DIR=/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/tables`
- `GENERATED_ARTIFACT=/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979/generated_subgoal_acting_d50_hh2_lh2_lrh1.npz`
- `ONLINE_ARTIFACT=/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979/online_hierarchical_logging_d50_hh2_lh2_lrh1.npz`
- `OUTPUT_DIR=/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual`

That manual directory exists and contains both figures and tables.

## Summary Tables Currently In The Repo Logs

### Offline summaries

Source files:

- [`logs/offline/summary_dataset_subgoal_reachability.tsv`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/logs/offline/summary_dataset_subgoal_reachability.tsv)
- [`logs/offline/summary_generated_subgoal_reachability.tsv`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/logs/offline/summary_generated_subgoal_reachability.tsv)
- [`logs/offline/summary_macro_manifold.tsv`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/logs/offline/summary_macro_manifold.tsv)
- [`logs/offline/summary_teacher_vs_open_loop.tsv`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/logs/offline/summary_teacher_vs_open_loop.tsv)

Key values:

```text
dataset_subgoal_reachability
D25 L1 overall_terminal_error_mean=0.1752391625
D50 L1 overall_terminal_error_mean=0.1554233087
D25 L2 overall_terminal_error_mean=0.0578273522
D25 L3 overall_terminal_error_mean=0.0351296014
D50 L2 overall_terminal_error_mean=0.0397784413
D50 L3 overall_terminal_error_mean=0.0201272693
D25 L5 overall_terminal_error_mean=0.0332028229
D50 L5 overall_terminal_error_mean=0.0201749442

generated_subgoal_reachability
D25 H1/L2 step1_terminal_error_mean=0.1612268388 dataset_distance_mean=4.9989662170 same_traj_distance_mean=1.2643420696
D25 H2/L2 step1_terminal_error_mean=0.0119003309 step2_terminal_error_mean=0.0176116675
D50 H1/L2 step1_terminal_error_mean=0.5212045908 dataset_distance_mean=5.7064461708 same_traj_distance_mean=2.9161150455
D50 H2/L2 step1_terminal_error_mean=0.1113949791 step2_terminal_error_mean=0.1202083230

teacher_vs_open_loop
D25 H1 teacher=0.1176709533 open_true=0.1176709533 open_cem=0.0110389646 prior_issue=False instability=False
D25 H2 teacher=0.0281027108 open_true=0.0465629995 open_cem=0.0880894065 prior_issue=True instability=True
D50 H1 teacher=0.5377691984 open_true=0.5377691984 open_cem=0.0599851012 prior_issue=False instability=False
D50 H2 teacher=0.0924003944 open_true=0.1485966593 open_cem=0.1786198169 prior_issue=False instability=True
```

### Acting summaries

Source files:

- [`logs/acting/summary_generated_subgoal_acting.tsv`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/logs/acting/summary_generated_subgoal_acting.tsv)
- [`logs/acting/summary_low_level_reality_gap.tsv`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/logs/acting/summary_low_level_reality_gap.tsv)
- [`logs/acting/summary_online_hierarchical_logging.tsv`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/logs/acting/summary_online_hierarchical_logging.tsv)
- [`logs/acting/summary_oracle_subgoal_acting.tsv`](/gpfs/home2/scur0200/main/jobs/eval/hi/paper_diagnostics/logs/acting/summary_oracle_subgoal_acting.tsv)

Key values:

```text
oracle_subgoal_acting
D50 H2/L2/R1 success_rate=68.0 final_terminal_latent_error_mean=0.3044882119 reality_gap_mean=0.0461610854
D50 H2/L3/R1 success_rate=50.0 final_terminal_latent_error_mean=0.3440113068 reality_gap_mean=0.0359631858
D50 H2/L5/R1 success_rate=24.0 final_terminal_latent_error_mean=0.5642428398 reality_gap_mean=0.0543776232

generated_subgoal_acting
D50 H1/L2/R1 success_rate=32.0 step1_offset_error_token_mean=-2.4440002441 final_terminal_latent_error_mean=0.9980964661
D50 H2/L2/R1 success_rate=42.0 step1_offset_error_token_mean=-1.0920000076 step2_offset_error_token_mean=-0.6840000153 final_terminal_latent_error_mean=0.8310787678

low_level_reality_gap
L2/R1 overall_actual_error_mean=0.0859023941 overall_reality_gap_mean=0.0158283208
L3/R1 overall_actual_error_mean=0.1466468995 overall_reality_gap_mean=0.0116838767
L5/R1 overall_actual_error_mean=0.2993787279 overall_reality_gap_mean=0.0143454124

online_hierarchical_logging
H1/L2/R1 success_rate=44.0 final_terminal_latent_error_mean=0.9287943840 churn=0.0625241523 reality_gap=0.1097075634
H2/L5/R5 success_rate=12.0 final_terminal_latent_error_mean=1.0043604374 churn=0.3378441847 reality_gap=0.3798063248
H2/L2/R1 success_rate=30.0 final_terminal_latent_error_mean=0.8998420238 churn=0.3353749944 reality_gap=0.1160770714
H2/L5/R1 success_rate=16.0 final_terminal_latent_error_mean=1.0851762295 churn=0.3419745962 reality_gap=0.0629368968
```

## Scratch Output Bundle 1: Paper Diagnostics Report

Directory:

- [`/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474)

Manifest:

- [`manifest.json`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/manifest.json)

Tables:

- [`tables/generated_subgoal_acting.csv`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/tables/generated_subgoal_acting.csv)
- [`tables/low_level_reality_gap.csv`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/tables/low_level_reality_gap.csv)
- [`tables/online_hierarchical_logging.csv`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/tables/online_hierarchical_logging.csv)
- [`tables/oracle_subgoal_acting.csv`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/tables/oracle_subgoal_acting.csv)
- [`tables/teacher_vs_open_loop.csv`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/tables/teacher_vs_open_loop.csv)

Images:

- [`figures/failure_ladder_d50.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/failure_ladder_d50.png), 2359x1308
- [`figures/failure_ladder_d50.pdf`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/failure_ladder_d50.pdf)
- [`figures/generated_subgoal_offset_error.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/generated_subgoal_offset_error.png), 2209x1308
- [`figures/generated_subgoal_offset_error.pdf`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/generated_subgoal_offset_error.pdf)
- [`figures/low_level_horizon_sweep.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/low_level_horizon_sweep.png), 2959x1216
- [`figures/low_level_horizon_sweep.pdf`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/low_level_horizon_sweep.pdf)
- [`figures/online_churn_and_success.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/online_churn_and_success.png), 3259x1308
- [`figures/online_churn_and_success.pdf`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/online_churn_and_success.pdf)
- [`figures/teacher_vs_open_loop.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/teacher_vs_open_loop.png), 2659x1308
- [`figures/teacher_vs_open_loop.pdf`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/teacher_vs_open_loop.pdf)

What the visible paper figures show:

- `failure_ladder_d50`: bar chart with `Oracle subgoal acting=68`, `Generated subgoal acting=42`, `Online hierarchical=30`, `Baseline best=54`.
- `teacher_vs_open_loop`: grouped bar chart over `D25 H1`, `D25 H2`, `D50 H1`, `D50 H2`, comparing `Teacher-forced`, `Open-loop true macro`, and `Open-loop CEM macro`.
- `online_churn_and_success`: two side-by-side bar charts, one for success over `H1/L2/R1`, `H2/L2/R1`, `H2/L5/R1`, `H2/L5/R5`, and one for replanning instability via subgoal churn MSE over the same settings.

The paper tables in this bundle are numerically the same metrics already reflected in the repo-local summary TSVs.

## Scratch Output Bundle 2: Decoder Story Artifacts

Directory:

- [`/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979)

Files:

- [`generated_subgoal_acting_d50_hh2_lh2_lrh1.json`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979/generated_subgoal_acting_d50_hh2_lh2_lrh1.json)
- [`generated_subgoal_acting_d50_hh2_lh2_lrh1.npz`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979/generated_subgoal_acting_d50_hh2_lh2_lrh1.npz)
- [`online_hierarchical_logging_d50_hh2_lh2_lrh1.json`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979/online_hierarchical_logging_d50_hh2_lh2_lrh1.json)
- [`online_hierarchical_logging_d50_hh2_lh2_lrh1.npz`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979/online_hierarchical_logging_d50_hh2_lh2_lrh1.npz)
- [`summary_generated_subgoal_acting.tsv`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979/summary_generated_subgoal_acting.tsv)
- [`summary_online_hierarchical_logging.tsv`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_artifacts_hope2_23062979/summary_online_hierarchical_logging.tsv)

Top-level stats in these artifact summaries:

```text
generated_subgoal_acting_d50_hh2_lh2_lrh1
step1_stage_end_actual_error_mean=0.4359828234
step1_offset_error_token_mean=-1.0920000076
step2_stage_end_actual_error_mean=0.6226748824
step2_offset_error_token_mean=-0.6840000153
final_terminal_latent_error_mean=0.8172678351
success_rate=42.0

online_hierarchical_logging_d50_hh2_lh2_lrh1
success_rate=34.0
final_terminal_latent_error_mean=0.8930814266
mean_distance_current_to_subgoal=0.4969123900
mean_distance_current_to_final_goal=1.9623848271
mean_subgoal_churn_mse=0.3372902771
mean_reality_gap=0.1124108516
```

These `.json` files are much richer than the TSVs:

- `generated_subgoal_acting...json` includes `step_results`, `stage_end_events`, and a long `low_block_events` list with per-block model error, actual error, reality gap, goal error, and selected-action statistics.
- `online_hierarchical_logging...json` includes `episode_successes` and `high_plan_events` with per-plan subgoal norm, churn, macro norms, and Mahalanobis statistics.

## Scratch Output Bundle 3: Decoder Story Figures

Directory:

- [`/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980)

Manifest:

- [`manifest.json`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/manifest.json)

Teacher/open-loop story images:

- [`teacher_vs_open_loop/teacher_story_00.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/teacher_vs_open_loop/teacher_story_00.png), sample `241`, 1574x3097
- [`teacher_vs_open_loop/teacher_story_01.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/teacher_vs_open_loop/teacher_story_01.png), sample `184`, 1574x3097
- [`teacher_vs_open_loop/teacher_story_02.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/teacher_vs_open_loop/teacher_story_02.png), sample `213`, 1574x3097
- [`teacher_vs_open_loop/teacher_story_03.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/teacher_vs_open_loop/teacher_story_03.png), sample `6`, 1504x3097

Generated-subgoal story images:

- [`generated_subgoal_acting/generated_story_00.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/generated_subgoal_acting/generated_story_00.png), sample `19`, `success=false`, `stage1_offset_error_steps=-24`, `final_error_mean=2.3146069050`, 3440x2803
- [`generated_subgoal_acting/generated_story_01.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/generated_subgoal_acting/generated_story_01.png), sample `20`, `success=false`, `stage1_offset_error_steps=-24`, `final_error_mean=2.2564206123`, 3440x2803
- [`generated_subgoal_acting/generated_story_02.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/generated_subgoal_acting/generated_story_02.png), sample `9`, `success=false`, `stage1_offset_error_steps=-24`, `final_error_mean=2.1234529018`, 3440x2803
- [`generated_subgoal_acting/generated_story_03.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/generated_subgoal_acting/generated_story_03.png), sample `45`, `success=false`, `stage1_offset_error_steps=-24`, `final_error_mean=1.8648868799`, 3440x2803
- [`generated_subgoal_acting/generated_story_04.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/generated_subgoal_acting/generated_story_04.png), sample `26`, `success=false`, `stage1_offset_error_steps=-24`, `final_error_mean=1.8043187857`, 3440x2803
- [`generated_subgoal_acting/generated_story_05.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/generated_subgoal_acting/generated_story_05.png), sample `36`, `success=false`, `stage1_offset_error_steps=-24`, `final_error_mean=1.7355064154`, 3440x2803

Online replanning story images:

- [`online_hierarchical_logging/online_story_00.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/online_hierarchical_logging/online_story_00.png), sample `26`, `success=false`, `final_error_mean=2.2971103191`, `num_high_plans=10`, 10697x2303
- [`online_hierarchical_logging/online_story_01.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/online_hierarchical_logging/online_story_01.png), sample `20`, `success=false`, `final_error_mean=2.2449903488`, `num_high_plans=10`, 10697x2303
- [`online_hierarchical_logging/online_story_02.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/online_hierarchical_logging/online_story_02.png), sample `22`, `success=false`, `final_error_mean=2.1144006252`, `num_high_plans=10`, 10697x2303
- [`online_hierarchical_logging/online_story_03.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/online_hierarchical_logging/online_story_03.png), sample `18`, `success=false`, `final_error_mean=2.1102626324`, `num_high_plans=10`, 10697x2303

Visual content of these images:

- `teacher_story_*`: each figure stacks context, target, teacher rollout, open-loop true rollout, and open-loop CEM rollout for one selected episode and labels the teacher/CEM errors.
- `generated_story_*`: each figure shows a failed staged plan; the visible example `generated_story_00` is titled `Generated Subgoal Acting Case 1 (fail) | stage1 offset -24 steps`.
- `online_story_*`: each figure shows repeated high-level replans over an episode; the visible example `online_story_00` is titled `Online Replanning Case 1 (fail)` and spans `Plan 1` through `Plan 10`, then `Goal` and `Final`.

## Scratch Output Bundle 4: Manual Story-Figure Bundle

Directory:

- [`/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual)

Manifest:

- [`manifest.json`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/manifest.json)

Figures:

- [`figures/probe/probe_sanity.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/figures/probe/probe_sanity.png), 3262x2753
- [`figures/offline/offline_forecast_stage_1.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/figures/offline/offline_forecast_stage_1.png), 3267x2744
- [`figures/offline/offline_forecast_stage_2.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/figures/offline/offline_forecast_stage_2.png), 3267x2744
- [`figures/acting/oracle_vs_generated.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/figures/acting/oracle_vs_generated.png), 4587x2748
- [`figures/acting/temporal_validity.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/figures/acting/temporal_validity.png), 3927x2746
- [`figures/online/online_replanning.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/figures/online/online_replanning.png), 3925x2750

Tables:

- [`tables/failure_decomposition.csv`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/tables/failure_decomposition.csv)
- [`tables/generated_subgoal_validity.csv`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/tables/generated_subgoal_validity.csv)
- [`tables/high_level_forecast.csv`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/tables/high_level_forecast.csv)
- [`tables/online_replanning.csv`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/tables/online_replanning.csv)
- [`tables/open_loop_cem_compounding.csv`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/tables/open_loop_cem_compounding.csv)
- [`tables/open_loop_error_dynamics.csv`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/tables/open_loop_error_dynamics.csv)
- [`tables/open_loop_method_comparison.csv`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/tables/open_loop_method_comparison.csv)
- [`tables/story_tables.md`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/tables/story_tables.md)

Selected exact table contents:

```text
failure_decomposition
baseline_best_d50,54.0
oracle_subgoal_acting,68.0,0.3044882118701935,0.0461610853672027
generated_subgoal_acting,42.0,0.8310787677764893
online_hierarchical,30.0,0.8998420238494873,0.1160770714282989,0.3353749944104088

generated_subgoal_validity
D50_H1,-2.444000244140625,,32.0
D50_H2,-1.0920000076293943,-0.6840000152587891,42.0

high_level_forecast
D25_H1,0.1176709532737732,0.1176709532737732,0.0110389646142721
D25_H2,0.0281027108430862,0.0465629994869232,0.0880894064903259
D50_H1,0.5377691984176636,0.5377691984176636,0.0599851012229919
D50_H2,0.0924003943800926,0.1485966593027115,0.1786198168992996

online_replanning
D50_H1_L2_R1,44.0,0.0625241522987683,0.1097075633704662
D50_H2_L5_R5,12.0,0.3378441847032971,0.379806324839592
D50_H2_L2_R1,30.0,0.3353749944104088,0.1160770714282989
D50_H2_L5_R1,16.0,0.3419745961825053,0.0629368968307972
```

What these figures visibly show:

- `offline_forecast_stage_1`: panelized context/target/teacher/open-loop-true/open-loop-CEM comparison for several selected episodes at stage 1.
- `oracle_vs_generated`: side-by-side oracle and generated subgoal acting outcomes for selected failed episodes.
- `online_replanning`: panelized online replanning instability examples with alternating planned goal and actually reached state panels.

## Per-Job Log Status

Counts under repo-local `logs/`:

- offline `.out` files: `20`
- offline `.err` files: `20`
- acting `.out` files: `12`
- acting `.err` files: `12`

Observed status:

- All checked `.out` logs report `result_status: "ok"` in their summaries.
- A grep over `jobs/eval/hi/paper_diagnostics/logs` and `jobs/eval/hi/paper_diagnostics/output` found no `Traceback`, `Exception`, `RuntimeError`, `ValueError`, `AssertionError`, or `ERROR:` lines.
- The `.err` files are non-empty, but they contain warnings rather than hard failures.

Recurring stderr warnings:

- SURF warning about mixing module and conda environments.
- `pygame` warning that `pkg_resources` is deprecated.
- `gymnasium` warning about casting to numpy arrays.
- `gymnasium` passive-env-checker warning that some returned observations are not within the observation space.
- `torch` warning that nested tensors are still prototype-stage.
- decoder-story render stderr also shows `torch.cuda` warning `Can't initialize NVML`.
- decoder-story render stderr also shows Matplotlib `tight_layout` warning from `scripts/diagnostics/render_hi_decoder_diagnostic_stories.py`.

## Short Interpretation Snapshot

The current artifacts support the same failure-analysis story you already wrote:

- oracle staged subgoals at `D50/H2/L2/R1` reach `68%`, above the baseline-best `54%`
- generated staged subgoals drop to `42%`
- online hierarchical replanning drops further to `30%`
- the D50 offline forecast numbers show that `H2` is the regime where open-loop error grows and instability becomes visible
- the story figures make the qualitative failure modes concrete: temporal misplacement of generated subgoals, weak final reachability after generated staging, and high online subgoal churn

## Download-Oriented Image List

If you only want the downloadable PNG outputs, these are the main ones:

### Main report PNGs

- [`failure_ladder_d50.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/failure_ladder_d50.png)
- [`generated_subgoal_offset_error.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/generated_subgoal_offset_error.png)
- [`low_level_horizon_sweep.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/low_level_horizon_sweep.png)
- [`online_churn_and_success.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/online_churn_and_success.png)
- [`teacher_vs_open_loop.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_paper_diagnostics_hope2_23048474/figures/teacher_vs_open_loop.png)

### Decoder-story PNGs

- [`teacher_story_00.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/teacher_vs_open_loop/teacher_story_00.png)
- [`teacher_story_01.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/teacher_vs_open_loop/teacher_story_01.png)
- [`teacher_story_02.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/teacher_vs_open_loop/teacher_story_02.png)
- [`teacher_story_03.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/teacher_vs_open_loop/teacher_story_03.png)
- [`generated_story_00.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/generated_subgoal_acting/generated_story_00.png)
- [`generated_story_01.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/generated_subgoal_acting/generated_story_01.png)
- [`generated_story_02.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/generated_subgoal_acting/generated_story_02.png)
- [`generated_story_03.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/generated_subgoal_acting/generated_story_03.png)
- [`generated_story_04.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/generated_subgoal_acting/generated_story_04.png)
- [`generated_story_05.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/generated_subgoal_acting/generated_story_05.png)
- [`online_story_00.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/online_hierarchical_logging/online_story_00.png)
- [`online_story_01.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/online_hierarchical_logging/online_story_01.png)
- [`online_story_02.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/online_hierarchical_logging/online_story_02.png)
- [`online_story_03.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_decoder_story_figures_hope2_23062980/online_hierarchical_logging/online_story_03.png)

### Manual story-bundle PNGs

- [`probe_sanity.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/figures/probe/probe_sanity.png)
- [`offline_forecast_stage_1.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/figures/offline/offline_forecast_stage_1.png)
- [`offline_forecast_stage_2.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/figures/offline/offline_forecast_stage_2.png)
- [`oracle_vs_generated.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/figures/acting/oracle_vs_generated.png)
- [`temporal_validity.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/figures/acting/temporal_validity.png)
- [`online_replanning.png`](/scratch-shared/scur0200/stablewm_data/reports/hi_story_figures_hope2_manual/figures/online/online_replanning.png)
