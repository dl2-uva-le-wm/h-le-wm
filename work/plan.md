```
Proposed Code Changes: Base LeWM → Hi-LeWM
The new codebase lives in hi-le-wm/ and copies all files from le-wm/, modifying the 5 files below. Nothing in stable_worldmodel or stable_pretraining is touched.

File 1: module.py — three new classes
1a. InverseDynamicsModel
Both ID^(3) and ID^(2) are the same architecture. It receives the concatenation of two latent states and outputs a macro-action vector.


Input:  [z_t ; z_target] ∈ R^{2D}
Output: ã ∈ R^{macro_action_dim}

Architecture:
  Linear(2D → 2D) → LayerNorm → GELU
  Linear(2D → D)  → LayerNorm → GELU
  Linear(D → macro_action_dim)
This is exactly what the existing MLP class does, except we need two inputs. Concretely:


class InverseDynamicsModel(nn.Module):
    def __init__(self, embed_dim: int, macro_action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, macro_action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        # z_t, z_target: (B, D)  →  returns ã: (B, macro_action_dim)
        return self.net(torch.cat([z_t, z_target], dim=-1))
1b. ConditionedSingleStepPredictor
Used for both pred^(3) (Level 3) and pred^(2) (Level 2). A small conditioned Transformer that takes 1–2 latent tokens as context and a macro-action as AdaLN-zero conditioning, producing a single output latent.


Level 3 call:  pred3(z_t,       ã^(3))           → ẑ_{t+k2}
Level 2 call:  pred2(z_t, ẑ^(3)_{t+k2}, ã^(2))  → ẑ_{t+k1}
The design reuses the existing Transformer (with ConditionalBlock) — we just vary the sequence length (1 or 2 tokens):


class ConditionedSingleStepPredictor(nn.Module):
    """
    Single-step predictor for strategic (Level 3) and tactical (Level 2).
    Takes a short context [z_current] or [z_current, z_anchor] plus a macro-action
    as AdaLN-zero conditioning and predicts one future latent.
    """
    def __init__(self, embed_dim: int, macro_action_dim: int,
                 depth: int = 3, heads: int = 8, mlp_dim: int = 1024,
                 dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        # project macro-action to embedding space for AdaLN
        self.action_proj = nn.Linear(macro_action_dim, embed_dim)
        self.transformer = Transformer(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            block_class=ConditionalBlock,   # AdaLN-zero, exists in module.py
        )

    def forward(
        self,
        z_current: torch.Tensor,           # (B, D)
        macro_action: torch.Tensor,        # (B, macro_action_dim)
        z_anchor: torch.Tensor | None = None,  # (B, D) or None
    ) -> torch.Tensor:                     # → (B, D)

        # build input sequence: [z_current] or [z_current, z_anchor]
        tokens = z_current.unsqueeze(1)                # (B, 1, D)
        if z_anchor is not None:
            tokens = torch.cat([tokens, z_anchor.unsqueeze(1)], dim=1)  # (B, 2, D)

        # macro-action conditioning, broadcast across tokens
        c = self.action_proj(macro_action)             # (B, D)
        c = c.unsqueeze(1).expand_as(tokens)           # (B, 1or2, D)

        out = self.transformer(tokens, c)              # (B, 1or2, D)
        return out[:, 0]                               # take first token as prediction
1c. Modified ARPredictor for Level 1 (anchor-conditioned)
The existing ARPredictor does not know about the midpoint anchor. We add an optional z_anchor argument that is projected and added to the action-embedding conditioning signal — no structural change to ConditionalBlock needed.


class ARPredictorAnchored(ARPredictor):
    """
    Extends ARPredictor with optional midpoint-anchor conditioning.
    The anchor is projected to embed_dim and added to the action embedding
    at every time step, so every ConditionalBlock sees the midpoint constraint.
    """
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.anchor_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,          # (B, T, D)  — latent history
        c: torch.Tensor,          # (B, T, D)  — action embeddings
        z_anchor: torch.Tensor | None = None,  # (B, D) or None
    ) -> torch.Tensor:
        if z_anchor is not None:
            # broadcast anchor across T and add to conditioning
            anchor_emb = self.anchor_proj(z_anchor)          # (B, D)
            c = c + anchor_emb.unsqueeze(1).expand_as(c)     # (B, T, D)
        return super().forward(x, c)
File 2: hi_jepa.py — new file (replaces jepa.py)
This is the core of the extension. HiJEPA wraps all 5 learned components and defines the training forward pass and the hierarchical inference.


class HiJEPA(nn.Module):
    """Top-Down Temporal Hierarchy world model.
    Components:
      encoder    — shared ViT encoder  (frozen: same as base LeWM)
      pred1      — ARPredictorAnchored (reactive, Level 1)
      pred2      — ConditionedSingleStepPredictor (tactical, Level 2)
      pred3      — ConditionedSingleStepPredictor (strategic, Level 3)
      id2        — InverseDynamicsModel (tactical)
      id3        — InverseDynamicsModel (strategic)
      action_encoder, projector, pred_proj  — same as base LeWM
    """
Key methods:

encode(info) — identical to JEPA.encode, no change.

train_forward(emb, act_emb, k1, k2):


Inputs:
  emb:     (B, history_size + k2, D)  — all encoded embeddings
  act_emb: (B, history_size,      D)  — action embeddings for the context
  k1, k2:  int — temporal scales

Returns: dict with pred1_emb, pred2_emb, pred3_emb, ã2_vec, ã3_vec
Inside:


ctx_len = history_size
z_t   = emb[:, ctx_len - 1]       # current state (last context frame)
z_k1  = emb[:, ctx_len - 1 + k1]  # ground truth k1-ahead
z_k2  = emb[:, ctx_len - 1 + k2]  # ground truth k2-ahead

# Level 3 — strategic
ã3         = id3(z_t, z_k2)
ẑ3         = pred3(z_t, ã3)        # no anchor at Level 3

# Level 2 — tactical (teacher forcing: use GT z_k2 as anchor)
ã2         = id2(z_t, z_k1)
ẑ2         = pred2(z_t, ã2, z_anchor=z_k2)  # GT anchor during training

# Level 1 — reactive (same as base LeWM, but anchor-conditioned)
pred1_out  = pred1(emb[:, :ctx_len], act_emb[:, :ctx_len],
                   z_anchor=z_k1)   # GT midpoint as anchor during training
# pred1_out[:, -1] ≈ emb[:, ctx_len]  (next step)
rollout(info, action_sequence, history_size) — hierarchical inference version:


Phase 0: encode z_t, z_g
Phase 1: ã3 = id3(z_t, z_g);        ẑ3 = pred3(z_t, ã3)
Phase 2: ã2 = id2(z_t, ẑ3);         ẑ2 = pred2(z_t, ã2, z_anchor=ẑ3)
Phase 3: Level 1 autoregressive rollout over horizon k1
         anchored by ẑ2 (passes ẑ2 to ARPredictorAnchored)
criterion(info_dict) — modified cost function:


# Use distance to midpoint anchor (ẑ2) rather than distance to distant goal (z_g)
cost = MSE(pred_emb[:, -1], info_dict["midpoint_anchor"])
get_cost(info_dict, action_candidates) — same structure as JEPA.get_cost but calls the hierarchical rollout and the new criterion.

File 3: hi_train.py — forward function and model construction
3a. New forward function

def hi_lejepa_forward(self, batch, stage, cfg):
    k1    = cfg.wm.k1
    k2    = cfg.wm.k2
    α     = cfg.loss.act_reg.weight
    λ     = cfg.loss.sigreg.weight
    ctx   = cfg.wm.history_size

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)   # encodes ALL num_steps frames
    emb    = output["emb"]              # (B, ctx + k2, D)
    act_emb = output["act_emb"]         # (B, ctx + k2 - 1, D)  (actions between frames)

    # Extract anchor ground truths
    z_t  = emb[:, ctx - 1]
    z_k1 = emb[:, ctx - 1 + k1]
    z_k2 = emb[:, ctx - 1 + k2]

    # ---- Level 3: strategic ----
    ã3   = self.model.id3(z_t, z_k2)
    ẑ3   = self.model.pred3(z_t, ã3)
    L3   = F.mse_loss(ẑ3, z_k2.detach())

    # ---- Level 2: tactical (teacher-forced anchor) ----
    ã2   = self.model.id2(z_t, z_k1)
    ẑ2   = self.model.pred2(z_t, ã2, z_anchor=z_k2.detach())
    L2   = F.mse_loss(ẑ2, z_k1.detach())

    # ---- Level 1: reactive (same as base, but anchor-conditioned) ----
    pred1_out = self.model.predict(
        emb[:, :ctx],
        act_emb[:, :ctx],
        z_anchor=z_k1.detach()           # ARPredictorAnchored accepts this kwarg
    )
    L1 = F.mse_loss(pred1_out, emb[:, 1 : ctx + 1].detach())

    # ---- SIGReg: applied ONCE on base encoder output ----
    L_reg = self.sigreg(emb.transpose(0, 1))

    # ---- L2 penalty on macro-actions ----
    L_act = ã2.pow(2).mean() + ã3.pow(2).mean()

    loss = L1 + L2 + L3 + λ * L_reg + α * L_act

    output.update(dict(
        pred_loss_l1=L1, pred_loss_l2=L2, pred_loss_l3=L3,
        sigreg_loss=L_reg, act_reg_loss=L_act, loss=loss
    ))
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output
3b. Model construction changes
In the run() function, in addition to the base components, instantiate:


macro_action_dim = cfg.wm.macro_action_dim   # new hyperparameter, e.g. 64

id3 = InverseDynamicsModel(embed_dim, macro_action_dim)
id2 = InverseDynamicsModel(embed_dim, macro_action_dim)

pred3 = ConditionedSingleStepPredictor(
    embed_dim=embed_dim,
    macro_action_dim=macro_action_dim,
    **cfg.hi_predictor,   # depth, heads, etc.
)

pred2 = ConditionedSingleStepPredictor(
    embed_dim=embed_dim,
    macro_action_dim=macro_action_dim,
    **cfg.hi_predictor,
)

# pred1 is the modified ARPredictor
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
File 4: config/train/hi_lewm.yaml — new config
Inherit from lewm.yaml, add hierarchical parameters:


defaults:
  - lewm              # inherit all base settings
  - _self_

output_model_name: hi_lewm

# override num_steps: must accommodate k2 frames ahead
# num_steps = history_size + k2  (overrides the eval: formula in data/*.yaml)

wm:
  k1: 5              # tactical horizon (env steps, after frameskip)
  k2: 20             # strategic horizon
  macro_action_dim: 64

hi_predictor:        # architecture for pred2 and pred3
  depth: 3
  heads: 8
  mlp_dim: 1024
  dim_head: 64
  dropout: 0.1

loss:
  sigreg:
    weight: 0.09
    kwargs:
      knots: 17
      num_proj: 1024
  act_reg:
    weight: 0.01     # α — L2 penalty on macro-actions
The data/pusht.yaml override needed:


dataset:
  num_steps: ${eval:'${wm.history_size} + ${wm.k2}'}   # was: num_preds + history_size
File 5: hi_eval.py — hierarchical inference policy
The eval.py uses swm.policy.WorldModelPolicy which calls model.get_cost(info_dict, action_candidates) internally. The cleanest path is:

Override get_cost in HiJEPA to implement the 4-phase inference pipeline:

def get_cost(self, info_dict, action_candidates):
    """
    Hierarchical inference: Phases 0-4.
    action_candidates: (B, S, k1, action_dim)  — CEM samples over SHORT horizon k1
    """
    device = next(self.parameters()).device
    # move to device ...

    # Phase 0: encode start and goal
    goal = self.encode({"pixels": info_dict["goal"][:, 0]})
    z_g  = goal["emb"][:, 0]              # (B, D)
    start = self.encode({"pixels": info_dict["pixels"][:, 0, -1:]})
    z_t  = start["emb"][:, 0]             # (B, D)

    # Phase 1: Level 3 — strategic anchor (single forward pass)
    ã3   = self.id3(z_t, z_g)
    ẑ3   = self.pred3(z_t, ã3)            # (B, D)

    # Phase 2: Level 2 — tactical midpoint (single forward pass)
    ã2   = self.id2(z_t, ẑ3)
    ẑ2   = self.pred2(z_t, ã2, z_anchor=ẑ3)   # (B, D) — midpoint anchor

    # Phase 3: Level 1 — short-horizon CEM rollout
    # Store ẑ2 as "goal_emb" so criterion() compares against it
    info_dict["midpoint_anchor"] = ẑ2
    info_dict["goal_emb"]        = ẑ2     # reuse existing criterion machinery

    # Delegate to base rollout (horizon = k1, anchored by ẑ2)
    info_dict = self.rollout(info_dict, action_candidates)
    cost = self.criterion(info_dict)       # MSE to ẑ2 instead of z_g
    return cost
CEM horizon is now k1, not H. The plan_config.horizon in the eval YAML changes:

plan_config:
  horizon: 5          # was: full planning horizon H; now just k1
  receding_horizon: 1
  action_block: 5
The CEMSolver from stable_worldmodel is reused unchanged — it samples action_candidates of shape (B, S, horizon, action_dim) and calls model.get_cost. Since get_cost now uses the hierarchical phases internally, the solver never needs to know about the hierarchy.

Summary of new vs modified components
Component	Status	Where
InverseDynamicsModel	New	module.py
ConditionedSingleStepPredictor	New	module.py
ARPredictorAnchored	New (extends existing)	module.py
HiJEPA	New file	hi_jepa.py
hi_lejepa_forward	New function	hi_train.py
Model construction in run()	Modified	hi_train.py
config/train/hi_lewm.yaml	New	config/train/
config/train/data/*.yaml	num_steps override	each env config
HiJEPA.get_cost	Modified (replaces JEPA's)	hi_jepa.py
config/eval/hi_pusht.yaml	New	config/eval/
utils.py, stable_worldmodel, stable_pretraining	Untouched	—
Key design decisions to flag before implementation
Teacher forcing at Level 2: during training, pred2 receives the ground-truth z_{k2} as anchor (not ẑ3). This is more stable but creates a train/inference gap. If gap is a problem, you can schedule a curriculum from GT anchor → predicted anchor over training.

k1 and k2 as env steps (after frameskip): frameskip=5 in pusht, so k1=5 env steps = 1 frame in the dataset, k2=20 = 4 frames. The num_steps in the data config must count dataset frames, not env steps: num_steps = history_size + k2 // frameskip.

pred1 backward compatibility: ARPredictorAnchored.forward falls back gracefully (z_anchor=None) to exactly the base LeWM behavior, so you can test it with the same training loop as base LeWM first.

Macro-action dimension: macro_action_dim=64 (same as embed_dim=192 / 3) is a reasonable starting point. It's a new hyperparameter (Ablation 5 in your proposal tests sensitivity to k1/k2; similarly, sensitivity to macro_action_dim is worth an ablation).

SIGReg is computed on all frames in the sequence (base encoder output across history_size + k2 frames), not just the first history_size. This gives SIGReg a larger and more diverse batch of embeddings to regularize, which is strictly better.
```
