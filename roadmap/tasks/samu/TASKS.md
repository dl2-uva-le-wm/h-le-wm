# `samu` Tasks (Training + Config)

## Scope Ownership

- `hi_train.py` (new)
- `config/train/hi_lewm.yaml` (new)
- `config/train/data/*.yaml` (new `hi_` variants preferred)

## Pre-Flight Checks (Validate What Is Already Done)

Run:

```bash
test -f hi_train.py && echo "hi_train.py exists"
test -f config/train/hi_lewm.yaml && echo "hi_lewm.yaml exists"
python -m py_compile hi_train.py && echo "hi_train syntax ok"
rg -n "def hi_lejepa_forward|pred_loss_l1|pred_loss_l2|pred_loss_l3|act_reg_loss" hi_train.py
rg -n "k1_env|k2_env|macro_action_dim|k1_frames|k2_frames|anchor_mode" config/train/hi_lewm.yaml
```

Mark when verified:

- [ ] `hi_train.py` exists and compiles
- [ ] `hi_lewm.yaml` exists and contains key fields
- [ ] hierarchical loss keys present

## Task 1: Create `hi_train.py` Entry Point

Copy structure from `train.py`, then replace model + forward logic.

Implementation snippet (`hi_lejepa_forward`):

```python
def hi_lejepa_forward(self, batch, stage, cfg):
    ctx = cfg.wm.history_size
    k1 = cfg.wm.k1_frames
    k2 = cfg.wm.k2_frames
    lambd = cfg.loss.sigreg.weight
    alpha = cfg.loss.act_reg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]          # (B, ctx+k2, D)
    act_emb = output["act_emb"]  # (B, ctx+k2-1, D)

    z_t  = emb[:, ctx - 1]
    z_k1 = emb[:, ctx - 1 + k1]
    z_k2 = emb[:, ctx - 1 + k2]

    a3 = self.model.id3(z_t, z_k2)
    z3 = self.model.pred3(z_t, a3)
    L3 = F.mse_loss(z3, z_k2.detach())

    a2 = self.model.id2(z_t, z_k1)
    z2_anchor = z_k2.detach()
    z2 = self.model.pred2(z_t, a2, z_anchor=z2_anchor)
    L2 = F.mse_loss(z2, z_k1.detach())

    pred1 = self.model.predict(
        emb[:, :ctx],
        act_emb[:, :ctx],
        z_anchor=z_k1.detach(),
    )
    L1 = F.mse_loss(pred1, emb[:, 1 : ctx + 1].detach())

    L_reg = self.sigreg(emb.transpose(0, 1))
    L_act = a2.pow(2).mean() + a3.pow(2).mean()

    loss = L1 + L2 + L3 + lambd * L_reg + alpha * L_act

    output.update(
        pred_loss_l1=L1,
        pred_loss_l2=L2,
        pred_loss_l3=L3,
        sigreg_loss=L_reg,
        act_reg_loss=L_act,
        loss=loss,
    )
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output
```

Done criteria:

- one-batch forward works on GPU.
- no index-out-of-range when `ctx`, `k1_frames`, `k2_frames` are set.

## Task 2: Build Model Assembly in `hi_train.py`

Add imports:

```python
from hi_jepa import HiJEPA
from module import (
    ARPredictorAnchored,
    ConditionedSingleStepPredictor,
    InverseDynamicsModel,
    Embedder, MLP, SIGReg,
)
```

Instantiation snippet:

```python
macro_action_dim = cfg.wm.macro_action_dim

id3 = InverseDynamicsModel(embed_dim, macro_action_dim)
id2 = InverseDynamicsModel(embed_dim, macro_action_dim)

pred3 = ConditionedSingleStepPredictor(embed_dim, macro_action_dim, **cfg.hi_predictor)
pred2 = ConditionedSingleStepPredictor(embed_dim, macro_action_dim, **cfg.hi_predictor)

pred1 = ARPredictorAnchored(
    embed_dim=embed_dim,
    num_frames=cfg.wm.history_size,
    input_dim=embed_dim,
    hidden_dim=hidden_dim,
    output_dim=hidden_dim,
    **cfg.predictor,
)

world_model = HiJEPA(
    encoder=encoder,
    pred1=pred1, pred2=pred2, pred3=pred3,
    id2=id2, id3=id3,
    action_encoder=action_encoder,
    projector=projector,
    pred_proj=predictor_proj,
)
```

## Task 3: Add `config/train/hi_lewm.yaml`

Create file with this minimal block:

```yaml
defaults:
  - lewm
  - _self_

output_model_name: hi_lewm

wm:
  k1_env: 5
  k2_env: 20
  macro_action_dim: 64
  k1_frames: ${eval:'${wm.k1_env} // ${data.dataset.frameskip}'}
  k2_frames: ${eval:'${wm.k2_env} // ${data.dataset.frameskip}'}

hi_predictor:
  depth: 3
  heads: 8
  mlp_dim: 1024
  dim_head: 64
  dropout: 0.1

loss:
  act_reg:
    weight: 0.01
  sigreg:
    weight: 0.09
    kwargs:
      knots: 17
      num_proj: 1024

anchor_mode: gt_only
```

## Task 4: Add `hi_` Data Config (Preferred)

Create `config/train/data/hi_pusht.yaml`:

```yaml
defaults:
  - pusht
  - _self_

dataset:
  num_steps: ${eval:'${wm.history_size} + ${wm.k2_frames}'}
```

If reusing `pusht.yaml`, still enforce this formula.

## Task 5: Teacher-Forcing Curriculum Switch

Add helper in `hi_train.py`:

```python
def pick_l2_anchor(cfg, z_k2_gt, z_k2_pred, step):
    mode = cfg.get("anchor_mode", "gt_only")
    if mode == "gt_only":
        return z_k2_gt.detach()
    if mode == "pred_only":
        return z_k2_pred.detach()
    # mode == "mix"
    warmup = max(int(cfg.get("anchor_mix_warmup_steps", 1)), 1)
    p_pred = min(1.0, step / warmup)
    use_pred = (torch.rand((), device=z_k2_gt.device) < p_pred).item()
    return z_k2_pred.detach() if use_pred else z_k2_gt.detach()
```

## Run Commands To Include In PR

```bash
python train.py
python hi_train.py
```

If using hydra overrides:

```bash
python hi_train.py data=hi_pusht output_model_name=hi_lewm_smoke trainer.max_epochs=1
```

## Handoff Artifact

- PR comment block:
  - exact launch command
  - resolved `k1_frames`, `k2_frames`
  - first 20 training-step losses.

Post-implementation checks:

```bash
python hi_train.py trainer.max_epochs=1 loader.batch_size=2
```

- [ ] one-epoch smoke run completes
- [ ] no index-out-of-range errors on `z_t`, `z_k1`, `z_k2`
- [ ] loss logs include L1/L2/L3 + reg terms
