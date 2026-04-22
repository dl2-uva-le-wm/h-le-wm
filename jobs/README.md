# Jobs Folder

This directory contains reproducible HPC job scripts for dataset setup, training, and evaluation.
It is organized to stay clean when moving work across branches (including the later move to `franco`).

## Layout

- `setup/`: environment bootstrap + dataset download/test jobs.
- `train/pusht/`: PushT training and benchmark jobs.
- `eval/original/`: baseline LeWM eval jobs.
- `eval/hi/`: hierarchical eval jobs and eval-specific docs.

## Versioning Policy

- Keep tracked:
  - `*.sh` job scripts
  - `*.md` documentation
- Do not track:
  - Slurm outputs (`*.out`, `*.err`)
  - Runtime logs (`*.log`)
  - sweep submission tables (`submitted_jobs.tsv`)

The ignore rules for these artifacts live in `jobs/.gitignore`.
