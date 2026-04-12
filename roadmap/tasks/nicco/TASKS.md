# `nicco` Tasks (Hierarchical Planner Integration)

## Scope Ownership

- `hi_jepa.py` (new)
- `config/eval/hi_*.yaml` (new)

## Coexistence & Non-Regression Policy

1. Do not break base `jepa.py` planning behavior.
2. Keep all hierarchical logic in `hi_jepa.py` and `hi_*.yaml`.
3. Before merge, verify:
   - base eval config still runs
   - hierarchical eval config runs.

## Pre-Flight Checks (Validate What Is Already Done)

Run these checks before new edits:

1. File exists and syntax is valid:

```bash
test -f hi_jepa.py && echo "hi_jepa.py exists"
python -m py_compile hi_jepa.py && echo "syntax ok"
```

2. `HiJEPA` core methods are present:

```bash
rg -n "class HiJEPA|def encode|def predict|def rollout|def criterion|def get_cost" hi_jepa.py
```

3. 4-phase planning markers exist in `get_cost`:

```bash
rg -n "Phase 0|Phase 1|Phase 2|Phase 3" hi_jepa.py
```

4. Midpoint anchor path is wired:

```bash
rg -n "midpoint_anchor|strategic_anchor|_compute_anchors" hi_jepa.py
```

5. Shape comments/docstrings are present:

```bash
rg -n "Args:|Returns:|\\(B, S|\\(B, D|\\(B, T" hi_jepa.py
```

If all pass, mark:

- [ ] `hi_jepa.py` implemented
- [ ] syntax check passed
- [ ] hierarchical phases present
- [ ] midpoint anchor wiring verified
- [ ] shape docs/comments present

## Task 1: Create `HiJEPA` Class Skeleton

Start from `jepa.py` but rename class and wire hierarchical components.

Constructor snippet:

```python
class HiJEPA(nn.Module):
    def __init__(
        self,
        encoder,
        pred1,
        pred2,
        pred3,
        id2,
        id3,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.pred1 = pred1
        self.pred2 = pred2
        self.pred3 = pred3
        self.id2 = id2
        self.id3 = id3
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
```

`encode` snippet:

```python
def encode(self, info):
    pixels = info["pixels"].float()
    b = pixels.size(0)
    pixels = rearrange(pixels, "b t ... -> (b t) ...")
    out = self.encoder(pixels, interpolate_pos_encoding=True)
    cls = out.last_hidden_state[:, 0]
    emb = self.projector(cls)
    info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
    if "action" in info:
        info["act_emb"] = self.action_encoder(info["action"])
    return info
```

## Task 2: Implement Level-1 `predict` + `rollout`

Predict wrapper snippet:

```python
def predict(self, emb, act_emb, z_anchor=None):
    preds = self.pred1(emb, act_emb, z_anchor=z_anchor)
    preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
    preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
    return preds
```

Rollout contract:

1. Input candidates shape `(B,S,H,action_dim)`.
2. Flatten `(B,S)` to `(BS)`.
3. Keep history truncation exactly like base JEPA.
4. At each step call `predict(..., z_anchor=midpoint_anchor)`.

## Task 3: Implement Hierarchical `get_cost` (4 Phases)

Core snippet:

```python
def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
    assert "goal" in info_dict, "goal not in info_dict"
    device = next(self.parameters()).device
    for k, v in list(info_dict.items()):
        if torch.is_tensor(v):
            info_dict[k] = v.to(device)

    # Phase 0: encode start and goal
    goal = self.encode({"pixels": info_dict["goal"][:, 0]})
    z_g = goal["emb"][:, 0]
    start = self.encode({"pixels": info_dict["pixels"][:, 0, -1:]})
    z_t = start["emb"][:, 0]

    # Phase 1: strategic anchor
    a3 = self.id3(z_t, z_g)
    z3 = self.pred3(z_t, a3)

    # Phase 2: tactical midpoint
    a2 = self.id2(z_t, z3)
    z2 = self.pred2(z_t, a2, z_anchor=z3)

    # Phase 3: L1 rollout against midpoint anchor
    info_dict["midpoint_anchor"] = z2
    info_dict["goal_emb"] = z2
    info_dict = self.rollout(info_dict, action_candidates)
    return self.criterion(info_dict)
```

Criterion snippet:

```python
def criterion(self, info_dict):
    pred_emb = info_dict["predicted_emb"]  # (B,S,T,D)
    mid = info_dict["midpoint_anchor"]     # (B,D)
    target = mid.unsqueeze(1).unsqueeze(2).expand_as(pred_emb[..., -1:, :])
    cost = F.mse_loss(pred_emb[..., -1:, :], target.detach(), reduction="none")
    return cost.sum(dim=tuple(range(2, cost.ndim)))  # (B,S)
```

## Task 4: Add Eval Configs

Create `config/eval/hi_pusht.yaml` snippet:

```yaml
defaults:
  - pusht
  - _self_

policy: hi_lewm

plan_config:
  horizon: 1 # set to k1_frames for this dataset/config
  receding_horizon: 1
  action_block: 5

output:
  filename: hi_pusht_results.txt
```

Then add `hi_tworoom.yaml` and `hi_reacher.yaml` similarly.

## Task 5: Smoke Run

Run:

```bash
python eval.py +config_name=hi_pusht
```

or equivalent hydra override command used in this repo.

Post-run checks:

```bash
rg -n "predicted_emb|midpoint_anchor|strategic_anchor" hi_jepa.py
```

- [ ] evaluation command runs without crash
- [ ] output results file created
- [ ] no tensor shape/device mismatch errors

## Handoff Artifact

Create `roadmap/tasks/nico/runbook.md` with:

1. config files added
2. exact eval command per config
3. expected output path per run.
