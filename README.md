# h-le-wm

Hierarchical LeWorldModel workspace with a **frozen upstream baseline** and local hierarchical extensions.

## Repo Model

This repo is intentionally split:

- Baseline LeWM (read-only): `third_party/lewm` (git submodule, pinned commit)
- Hierarchical code (editable): root `hi_*` files + `config/*/hi_*.yaml`

Root wrappers keep the CLI simple:

- `python train.py ...` -> runs baseline training in `third_party/lewm`
- `python eval.py ...` -> runs baseline evaluation in `third_party/lewm`
- `python hi_train.py ...` -> runs hierarchical training (local)
- `python hi_eval.py ...` -> runs hierarchical evaluation (local configs)

## Setup

```bash
git clone https://github.com/NiccoloCase/h-le-wm.git
cd h-le-wm
git submodule update --init --recursive

uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Datasets

Use the helper script (recommended):

```bash
source scripts/setup_datasets.sh --datasets pusht,tworooms,reacher,cube
```

Or set a custom data root:

```bash
source scripts/setup_datasets.sh --home /absolute/path/to/stablewm_data --datasets pusht
```

## Baseline Commands (Frozen Upstream)

Train baseline LeWM:

```bash
python train.py data=pusht
```

Evaluate baseline LeWM:

```bash
python eval.py --config-name=pusht policy=pusht/lewm
```

You can pass any Hydra overrides through wrappers exactly as usual.

## Hierarchical Commands

### Train

Default (3 levels, `hi_lewm` config):

```bash
python hi_train.py
```

2-level run (lighter):

```bash
python hi_train.py wm.num_levels=2 wm.k1=10 data=hi_pusht output_model_name=hi_lewm_l2
```

3-level run:

```bash
python hi_train.py wm.num_levels=3 wm.k1=10 wm.k2=30 data=hi_pusht output_model_name=hi_lewm_l3
```

### Evaluate

PushT:

```bash
python hi_eval.py --config-name=hi_pusht policy=pusht/hi_lewm
```

TwoRoom:

```bash
python hi_eval.py --config-name=hi_tworoom policy=tworoom/hi_lewm
```

Reacher:

```bash
python hi_eval.py --config-name=hi_reacher policy=reacher/hi_lewm
```

Example 2-level eval command:

```bash
python hi_eval.py --config-name=hi_pusht policy=pusht/hi_lewm wm.num_levels=2
```

## Integrity and Safety

Check baseline isolation:

```bash
python scripts/check_baseline_integrity.py
```

What it verifies:

- submodule exists and is configured
- submodule HEAD matches locked hash in `BASELINE_LOCK.md`
- submodule has no local tracked changes
- baseline-owned files are not reintroduced at repo root

See lock/pinning policy in `BASELINE_LOCK.md`.

## Updating Baseline (Explicit Only)

Only do this intentionally:

```bash
git -C third_party/lewm fetch origin
git -C third_party/lewm checkout <new_commit>
python scripts/check_baseline_integrity.py --allow-pointer-update
```

Then update `BASELINE_LOCK.md` with the new hash + rationale.

## Handy Wrapper Smoke (No Real Run)

```bash
LEWM_WRAPPER_DRY_RUN=1 python train.py data=pusht
LEWM_WRAPPER_DRY_RUN=1 python eval.py --config-name=pusht
LEWM_WRAPPER_DRY_RUN=1 python hi_eval.py --config-name=hi_pusht
```

These commands print delegated baseline calls without launching training/evaluation.
