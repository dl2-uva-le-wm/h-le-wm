# Task Split By Contributor Nickname

This folder contains independent implementation tracks derived from:

- `roadmap/dl2_detailed_implementation_proposal.md`

## Contributor Nicknames

- `nico` -> Niccolo Caselli
- `fra` -> Francesco Massafra
- `ippo` -> Ippokratis Pantelidis
- `samu` -> Samuele Punzo
- `salvo` -> Salvatore Lo Sardo

## Parallel Work Rules

1. Each contributor edits only files listed in their `TASKS.md` scope.
2. Do not change another contributor’s files without explicit sync.
3. Every PR must include:
   - run command(s)
   - config used
   - seed(s)
   - artifact path(s)

## Coexistence & Non-Regression Policy

1. Keep base LeWM path intact and runnable (`jepa.py`, `train.py`, base eval configs).
2. Implement hierarchy additively (`hi_jepa.py`, `hi_train.py`, `config/*/hi_*.yaml`).
3. Do not silently alter base behavior.
4. For integration PRs, include:
   - one base smoke check result
   - one hierarchical smoke check result.

## Interface Contract (Must Stay Stable)

1. `HiJEPA.get_cost(info_dict, action_candidates) -> (B,S)` tensor.
2. `action_candidates` shape: `(B,S,H,action_dim)`.
3. Config names:
   - train: `config/train/hi_lewm.yaml`
   - eval: `config/eval/hi_*.yaml`
4. `k1`/`k2` tracked in env steps and converted to frame offsets via `frameskip`.
