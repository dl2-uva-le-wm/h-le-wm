# `fra` Tasks (Level-1 CEM + MPC Validation)

## Scope Ownership

- `hi_jepa.py` (rollout/criterion validation only)
- `roadmap/tasks/fra/*`

## Coexistence & Non-Regression Policy

1. Validate hierarchical rollout without changing base planner logic.
2. Keep validation artifacts separate for base vs hierarchical runs.
3. Before merge, include comparison evidence for both paths.

## Pre-Flight Checks (Validate What Is Already Done)

Run:

```bash
test -f hi_jepa.py && echo "hi_jepa.py exists"
python -m py_compile hi_jepa.py && echo "hi_jepa syntax ok"
rg -n "def rollout|def criterion|get_cost|midpoint_anchor|strategic_anchor" hi_jepa.py
```

Mark when verified:

- [ ] rollout/criterion methods exist
- [ ] midpoint anchor path exists
- [ ] file compiles before validation scripts

## Task 1: Rollout Invariant Validation

Create `roadmap/tasks/fra/rollout_validation.py` and verify:

1. flatten/unflatten `(B,S)` is lossless
2. `history_size` truncation is always respected
3. midpoint anchor is broadcast correctly to `(BS,T,D)`

Validation snippet:

```python
def assert_rollout_shapes(info, out, B, S):
    pe = out["predicted_emb"]
    assert pe.ndim == 4, f"predicted_emb ndim expected 4, got {pe.ndim}"
    assert pe.shape[0] == B and pe.shape[1] == S, "Batch/sample mismatch after unflatten"

def assert_history_truncation(emb, HS):
    assert emb.shape[1] >= HS, "Embedding history shorter than history_size"
```

Test matrix:

- `(B,S,H)=(2,16,5)`
- `(B,S,H)=(4,32,5)`
- `(B,S,H)=(1,64,8)`

## Task 2: CEM/MPC Parameter Consistency

Create `roadmap/tasks/fra/cem_mpc_checks.md` with explicit checks:

1. `plan_config.horizon == k1_frames`
2. `receding_horizon <= horizon`
3. `action_block == frameskip` (or documented reason when not equal)

Log template:

```markdown
| env | horizon | k1_frames | receding_horizon | action_block | frameskip | ok |
|-----|---------|-----------|------------------|--------------|-----------|----|
| ... | ...     | ...       | ...              | ...          | ...       | Y/N|
```

## Task 3: Convergence Snapshot (Flat vs Hi)

Collect one controlled comparison:

1. same environment
2. same random seed
3. same CEM `num_samples` and `num_iters`

Output file:

- `roadmap/tasks/fra/convergence_snapshot.csv`

CSV schema:

```text
iteration,flat_best_cost,hi_best_cost
0,...
1,...
```

Plot snippet:

```python
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("roadmap/tasks/fra/convergence_snapshot.csv")
plt.plot(df["iteration"], df["flat_best_cost"], label="flat")
plt.plot(df["iteration"], df["hi_best_cost"], label="hi")
plt.legend(); plt.xlabel("iteration"); plt.ylabel("best cost")
plt.savefig("roadmap/tasks/fra/convergence_snapshot.png")
```

## Task 4: MPC Edge Cases

Add explicit notes for:

1. `eval_budget` close to horizon
2. `goal_offset_steps` shorter than expected
3. no-op or NaN actions from solver samples

Store in:

- `roadmap/tasks/fra/mpc_edge_cases.md`

## Handoff Artifact

- `roadmap/tasks/fra/invariants_checklist.md` with pass/fail for:
  - shape invariants
  - planner invariants
  - convergence sanity.

Post-validation checks:

```bash
test -f roadmap/tasks/fra/rollout_validation.py && echo "rollout_validation.py created"
test -f roadmap/tasks/fra/cem_mpc_checks.md && echo "cem_mpc_checks.md created"
test -f roadmap/tasks/fra/convergence_snapshot.csv && echo "convergence_snapshot.csv created"
```

- [ ] validation script produced expected checks
- [ ] CEM/MPC consistency table filled
- [ ] convergence artifact saved
