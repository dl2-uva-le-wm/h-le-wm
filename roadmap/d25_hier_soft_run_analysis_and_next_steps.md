# PushT d=25 Hierarchical Soft Run Analysis and Next Steps

## Run Reviewed
- Log: `jobs/eval/hi/eval_hope1_short_hier_soft_22005423.out`
- Result artifact: `/scratch-shared/scur0511/stablewm_data/runs/hi_lewm_p2_train_hope1_21983875/eval_hier_soft_d25_job_22005423/hi_lewm_p2_train_hope1_21983875_epoch_10_hi_pusht_results_d25_hier_soft.txt`
- Checkpoint: `epoch_10`
- Outcome: `success_rate = 40.0`

## What Improved vs Previous d=25 Hierarchical
- Previous hierarchical d=25 run was around `14%`.
- This run reached `40%` with:
  - high horizon reduced to `1`
  - low-level elite set changed (but to an extreme value, see below)

This indicates the short-horizon hierarchical path is sensitive to planner calibration and can be improved significantly by planning changes alone.

## Important Finding: Low-level `topk` Was Too High
From the run log command and echoed config, low-level planner used:
- `num_samples=300`
- `topk=150`

This means CEM kept 50% of candidates as elites each iteration.

### Why this likely hurts
CEM works best with selective elites. Keeping half the population reduces selection pressure, making updates too broad and less exploitative.

Typical effective ratios are around 5-15% elites. With `300` samples, that is usually `15-45` elites.

So `topk=150` likely prevented stronger convergence even though score improved from 14% to 40%.

## Additional Observations
- High-level solve time dropped to ~6.8s due to `high.horizon=1`, while low-level stayed ~30.6s.
- High-level latent prior calibration still ran and reported `chunks=2048`.
- The run completed cleanly and produced a valid results file.

## Recommended Next Config (Applied)
To keep the short-horizon hierarchical ablation but avoid overly broad elites, defaults were updated to a balanced stronger setup:

- High-level:
  - `num_samples=900` (unchanged)
  - `n_steps=20` (unchanged)
  - `topk=30` (was 10)
  - `horizon=1` (kept in short soft script)
  - `receding_horizon=1` (unchanged)
  - `replan_interval=5` (unchanged)

- Low-level:
  - `num_samples=600` (was 300)
  - `n_steps=30` (unchanged)
  - `topk=60` (was 150 in last run; now 10% elite ratio)
  - `horizon=5`, `receding_horizon=1`, `action_block=5` (unchanged)

## Files Updated
- `config/eval/hi_pusht.yaml`
- `jobs/eval/hi/eval_hope1_short_hier_soft.sh`

## Suggested Next Runs
Use same checkpoint for fair comparison:

1. Balanced short hierarchical (new default script):
```bash
sbatch --export=ALL,CHECKPOINT_EPOCH=10 jobs/eval/hi/eval_hope1_short_hier_soft.sh
```

2. If you want to isolate only `topk` effect at old sample count:
```bash
sbatch --export=ALL,CHECKPOINT_EPOCH=10,LOW_NUM_SAMPLES=300,LOW_TOPK=30,HIGH_TOPK=10 jobs/eval/hi/eval_hope1_short_hier_soft.sh
```

3. If run 1 improves but remains unstable, test more frequent macro replans:
```bash
sbatch --export=ALL,CHECKPOINT_EPOCH=10,HIGH_REPLAN_INTERVAL=3 jobs/eval/hi/eval_hope1_short_hier_soft.sh
```

## Interpretation Checklist for Next Result
- Did success exceed 40% with balanced elites?
- Do low-level solve times remain acceptable with 600 samples?
- Is variance across seeds reduced (if multiple seeds are run)?

If balanced elites improve score again, the main blocker is likely planner calibration/strength, not training collapse.
