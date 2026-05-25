# Hi Eval Jobs For Cube

This tree mirrors the hierarchical HOPE2 matrix workflow without touching the
active `jobs/eval/hi` PushT scripts.

## Layout

- `matrix/`: array-driven cube HOPE2 evaluation suite.

## Matrix Jobs

- `matrix/full_hierarchical_matrix_sweep.csv`
  - 39-row hierarchical planning sweep copied from the current HOPE2 matrix.
- `matrix/checkpoints_hope_hierarchical.txt`
  - Points at the finished cube HOPE2 checkpoint.
- `matrix/eval_hope_hierarchical_matrix.sh`
  - Array worker that maps one cube checkpoint row and one config row into a
    `python -m h_le_wm.eval.hierarchical --config-name=hi_cube` run.
- `matrix/submit_hope_hierarchical_matrix.sh`
  - Convenience submit helper that launches one 39-task array per checkpoint.
- `matrix/run_hi_cube_matrix_eval.sh`
  - Shared cube launcher used by the array worker.
