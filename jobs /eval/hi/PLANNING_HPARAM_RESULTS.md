# Planning Hyperparameter Results From Old `jobs/eval/hi` Slurm Logs

This document summarizes the effective planning hyperparameters recovered from the available `*.out` Slurm logs in `jobs/eval/hi/`, together with the reported environment `success_rate`.

Scope and conventions:

- `success_rate` is the main environment success metric printed by the run.
- Runs without an explicit `Eval budget:` line inherit `eval.eval_budget=50` from `config/eval/hi_pusht.yaml`.
- Runs without explicit planner summaries inherit defaults from `config/eval/hi_pusht.yaml`.
- The `block_only_success_rate` values in some logs were not used here because they appear clearly mis-scaled (`900.0`, `1000.0`).
- One run, `eval_hope1_short_hier_soft_lowerh_22063826.out`, did not finish with a printed `success_rate`, so it is listed as incomplete.

## Comparison Table

| Slurm out file | Ckpt | Mode | d | Budget | High planner | Low / flat planner | Success rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `eval_hope1_short_21995628.out` | latest (`epoch_10`) | hierarchical | 25 | 50 | `h=2, blk=1, rep=5, samp=900, iters=20, topk=30` | `h=5, blk=5, samp=600, iters=30, topk=60` | `14.0%` |
| `eval_hope1_medium_21995706.out` | latest (`epoch_10`) | hierarchical | 50 | 50 | `h=4, blk=1, rep=5, samp=1500, iters=40, topk=10` | `h=5, blk=5, samp=900, iters=20, topk=10` | `10.0%` |
| `eval_hope1_short_flat_22002409.out` | `8` | flat | 25 | 50 | `n/a` | `flat h=1, blk=5, samp=300, iters=30, topk=30` | `62.0%` |
| `eval_hope1_short_hier_soft_22003805.out` | `8` | hierarchical | 25 | 50 | `h=1, blk=1, rep=5, samp=900, iters=20, topk=10` | `h=5, blk=5, samp=300, iters=30, topk=30` | `26.0%` |
| `eval_hope1_short_hier_soft_22004751.out` | latest (`epoch_10`) | hierarchical | 25 | 50 | `h=1, blk=1, rep=5, samp=900, iters=20, topk=30` | `h=5, blk=5, samp=300, iters=30, topk=30` | `26.0%` |
| `eval_hope1_short_hier_soft_22005137.out` | latest (`epoch_10`) | hierarchical | 25 | 50 | `h=1, blk=1, rep=5, samp=900, iters=20, topk=10` | `h=5, blk=5, samp=300, iters=30, topk=50` | `34.0%` |
| `eval_hope1_short_hier_soft_22005423.out` | latest (`epoch_10`) | hierarchical | 25 | 50 | `h=1, blk=1, rep=5, samp=900, iters=20, topk=10` | `h=5, blk=5, samp=300, iters=30, topk=150` | `40.0%` |
| `eval_hope1_short_hier_soft_22005703.out` | latest (`epoch_10`) | hierarchical | 25 | 50 | `h=1, blk=1, rep=5, samp=900, iters=20, topk=30` | `h=5, blk=5, samp=600, iters=30, topk=60` | `28.0%` |
| `eval_hope1_short_hier_soft_22017859.out` | latest (`epoch_10`) | hierarchical | 25 | 75 | `h=1, blk=1, rep=5, samp=900, iters=20, topk=10` | `h=5, blk=5, samp=300, iters=30, topk=150` | `38.0%` |
| `eval_hope1_short_hier_soft_lowerbudg_22039237.out` | latest (`epoch_10`) | hierarchical | 25 | 30 | `h=1, blk=1, rep=5, samp=900, iters=20, topk=10` | `h=5, blk=5, samp=300, iters=30, topk=150` | `28.0%` |
| `eval_hope1_short_hier_soft_lowerh_22061755.out` | latest (`epoch_10`) | hierarchical | 25 | 50 | `h=1, blk=1, rep=5, samp=900, iters=20, topk=10` | `h=3, blk=5, samp=300, iters=30, topk=150` | `60.0%` |
| `eval_hope1_short_hier_soft_lowerh_22062865.out` | latest (`epoch_10`) | hierarchical | 25 | 50 | `h=1, blk=1, rep=5, samp=900, iters=20, topk=10` | `h=2, blk=5, samp=300, iters=30, topk=150` | `84.0%` |
| `eval_hope1_short_hier_soft_lowerh_22063826.out` | latest (`epoch_10`) | hierarchical | 25 | 30 | `h=1, blk=1, rep=5, samp=900, iters=20, topk=10` | `h=2, blk=5, samp=300, iters=30, topk=150` | `incomplete` |

## Quick Read

- Best completed run in these logs: `eval_hope1_short_hier_soft_lowerh_22062865.out` with `84.0%`.
- Strongest pattern: lowering the low-level horizon helped a lot.
  The same short hierarchical-soft family went from `40.0%` at `low h=5` to `60.0%` at `low h=3`, then to `84.0%` at `low h=2`.
- In the short hierarchical-soft family with `low h=5`, increasing low-level `topk` helped:
  `topk=30 -> 26.0%`, `topk=50 -> 34.0%`, `topk=150 -> 40.0%`.
- Reducing eval budget hurt for the same planner settings:
  `budget=50 -> 40.0%`, `budget=30 -> 28.0%`.
- Increasing eval budget to `75` for the same `low h=5, topk=150` family did not help versus the `50` budget run:
  `38.0%` vs `40.0%`.
- Flat planning on checkpoint `8` was strong at `62.0%`, better than most hierarchical settings except the `low h=3` and `low h=2` variants.

## Notes On Inherited Defaults

- `eval_hope1_short_21995628.out` does not print planner summaries, but its launch command only overrides `policy` and `eval.goal_offset_steps`.
  Its planner values therefore come from `config/eval/hi_pusht.yaml`:
  hierarchical mode, high `h=2/topk=30/samp=900/iters=20`, low `h=5/topk=60/samp=600/iters=30`, budget `50`.
- `eval_hope1_short_flat_22002409.out` does not print flat planner internals, but its launch command only forces `planning.mode=flat` and `eval.goal_offset_steps=25`.
  Its flat planner values therefore come from `config/eval/hi_pusht.yaml`:
  flat `h=1, blk=5, samp=300, iters=30, topk=30`, budget `50`.
- `eval_hope1_short_hier_soft_lowerh_22063826.out` prints the requested config, but the log ends before the final metrics block, so no success rate could be recovered from the `.out` file.
