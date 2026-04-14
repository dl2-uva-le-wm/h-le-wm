# Baseline Lock

This repository uses a **pinned baseline submodule** for the original LeWorldModel implementation.

## Source

- Repository: `https://github.com/lucas-maes/le-wm.git`
- Submodule path: `third_party/lewm`

## Pinned Commit

- Pinned Commit: `83f97d72ad067855bc89a1b74b4aff11d4dfdf0c`
- Commit title: `Initial commit`
- Commit date: `2026-03-12`

This commit is used as the paper-era frozen baseline.

## Policy

- Do not edit files inside `third_party/lewm` for feature development.
- Baseline upgrades must happen only by moving the submodule pointer in a dedicated PR.
- Root-level baseline wrappers (`train.py`, `eval.py`) delegate execution to this pinned submodule.
- Hierarchical development stays in root-level `hi_*` code and hierarchical config files.

## Update Procedure (Explicit Baseline Bump)

1. Update submodule pointer intentionally:
   - `git -C third_party/lewm fetch origin`
   - `git -C third_party/lewm checkout <new_commit>`
2. Update this file (`Pinned Commit`, title/date, rationale).
3. Run integrity + smoke checks:
   - `python scripts/check_baseline_integrity.py --allow-pointer-update`
4. Include baseline bump rationale in PR notes.

