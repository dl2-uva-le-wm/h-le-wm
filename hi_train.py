from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from hi_jepa import HiJEPA
from module import (
    ARPredictorAnchored,
    ConditionedSingleStepPredictor,
    Embedder,
    InverseDynamicsModel,
    MLP,
    SIGReg,
)
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


# ---------------------------------------------------------------------------
# Training forward pass
# ---------------------------------------------------------------------------

def hi_lejepa_forward(self, batch, stage, cfg):
    """One training/validation step for the Hierarchical LeWM.

    Encodes the full observation sequence, extracts ground-truth target
    embeddings at offsets +n_preds, +k1, and +k2 from the last context frame,
    then computes the joint hierarchical loss (proposal Eq. 16).

    Loss breakdown
    --------------
    l3_pred_loss : MSE( pred3(z_t, id3(z_t, z_{t+k2})),           z_{t+k2} )
    l2_pred_loss : MSE( pred2(z_t, id2(z_t, z_{t+k1}), z3_pred),  z_{t+k1} )
    l1_pred_loss : MSE( pred1_AR(ctx, act | z2_anchor),            tgt_l1   )
    sigreg_loss  : SIGReg applied once to shared encoder batch Z
    act_reg_loss : ||a2||^2 + ||a3||^2   (latent-action L2 penalty)

    Args:
        batch : dict — "pixels" (B, T, ...) and "action" (B, T, A_raw).
                T must satisfy T >= ctx_len + k2.
        stage : str  — "train" or "val" (metric log prefix).
        cfg   : OmegaConf node with wm.{history_size, num_preds, k1, k2}
                and loss.{sigreg, inverse_dynamics}.

    Returns:
        dict : output dict enriched with all computed loss tensors.
    """
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds           # AR step offset for Level-1 (typically 1)
    k1      = cfg.wm.k1                  # tactical horizon
    k2      = cfg.wm.k2                  # strategic horizon
    lambd   = cfg.loss.sigreg.weight
    alpha   = cfg.loss.inverse_dynamics.weight

    # Replace NaN actions at sequence boundaries with 0.
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # -- Encode the full sequence --------------------------------------------
    output  = self.model.encode(batch)
    emb     = output["emb"]       # (B, T, D)
    act_emb = output["act_emb"]   # (B, T, A)

    T     = emb.size(1)
    t_idx = ctx_len - 1           # index of the last context frame

    assert T >= ctx_len + k2, (
        f"Sequence length {T} is too short for ctx_len={ctx_len} + k2={k2}. "
        f"Set data.sequence_length >= {ctx_len + k2}."
    )

    # -- Ground-truth target embeddings -------------------------------------
    z_t   = emb[:, t_idx]                  # (B, D) — current latent state
    z_tk1 = emb[:, t_idx + k1]             # (B, D) — tactical target
    z_tk2 = emb[:, t_idx + k2]             # (B, D) — strategic target

    # -- Level 3 — Strategic Extrapolation ----------------------------------
    # ID3 infers the macro-action that bridges z_t -> z_{t+k2};
    # pred3 materialises the long-range anchor prediction.
    a3      = self.model.id3(z_t, z_tk2)   # (B, A3)
    z3_pred = self.model.pred3(z_t, a3)    # (B, D)
    output["l3_pred_loss"] = (z3_pred - z_tk2).pow(2).mean()

    # -- Level 2 — Tactical Interpolation -----------------------------------
    # ID2 infers the macro-action that bridges z_t -> z_{t+k1};
    # pred2 interpolates the midpoint, constrained by the detached L3 anchor
    # (mirrors inference-time behaviour where only z3_pred is available).
    a2      = self.model.id2(z_t, z_tk1)                              # (B, A2)
    z2_pred = self.model.pred2(z_t, a2, z_anchor=z3_pred.detach())    # (B, D)
    output["l2_pred_loss"] = (z2_pred - z_tk1).pow(2).mean()

    # -- Level 1 — Reactive Step Prediction ---------------------------------
    # AR predictor over the context window, globally conditioned on the
    # detached tactical midpoint anchor z2_pred.
    # AR supervision: pred[:, i] is trained to match emb[:, i + n_preds].
    ctx_emb = emb[:, :ctx_len]                        # (B, ctx_len, D)
    ctx_act = act_emb[:, :ctx_len]                    # (B, ctx_len, A)
    tgt_l1  = emb[:, n_preds : ctx_len + n_preds]     # (B, ctx_len, D)
    pred_l1 = self.model.predict(                     # (B, ctx_len, D)
        ctx_emb, ctx_act, z_anchor=z2_pred.detach()
    )
    output["l1_pred_loss"] = (pred_l1 - tgt_l1).pow(2).mean()

    # -- Anti-Collapse Regularization (once, shared encoder) ----------------
    # Applied exactly once — key simplification over bottom-up hierarchies.
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))  # (T, B, D)

    # -- Latent Action Regularization ---------------------------------------
    # Prevents ID models from degenerate shortcut solutions.
    #Notice that this is a BATCH mean so we divide also by D
    output["act_reg_loss"] = a2.pow(2).mean() + a3.pow(2).mean()

    # -- Total Objective (Eq. 16) -------------------------------------------
    output["loss"] = (
        output["l1_pred_loss"]
        + output["l2_pred_loss"]
        + output["l3_pred_loss"]
        + lambd * output["sigreg_loss"]
        + alpha * output["act_reg_loss"]
    )

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./config/train", config_name="hi_lewm")
def run(cfg):
    #########################
    ## dataset             ##
    #########################
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ## model construction       ##
    ##############################
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim        = encoder.config.hidden_size
    embed_dim         = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    # Macro-action dim shared by all ID models and single-step predictors.
    macro_action_dim  = cfg.wm.get("macro_action_dim", embed_dim)

    # -- Level 1: Anchored autoregressive predictor (reactive) ---------------
    # ARPredictorAnchored extends ARPredictor with an optional global z_anchor
    # that is added to the action-embedding condition c before each AR step.
    pred1 = ARPredictorAnchored(
        embed_dim=embed_dim,             # sizes the anchor projection layer
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,                 # depth, heads, mlp_dim, dim_head, …
    )

    # -- Level 2: Tactical single-step predictor ----------------------------
    # ConditionedSingleStepPredictor takes (z_current, macro_action, z_anchor)
    # and predicts one future latent via AdaLN-conditioned Transformer.
    pred_l2_cfg = cfg.get(
        "predictor_l2",
        {"depth": 3, "heads": 8, "mlp_dim": 1024, "dim_head": 64},
    )
    pred2 = ConditionedSingleStepPredictor(
        embed_dim=embed_dim,
        macro_action_dim=macro_action_dim,
        **pred_l2_cfg,
    )

    # -- Level 3: Strategic single-step predictor ---------------------------
    # Same architecture as pred2 but trained for the longer k2 horizon;
    # no higher-level anchor (top of the hierarchy).
    pred_l3_cfg = cfg.get(
        "predictor_l3",
        {"depth": 3, "heads": 8, "mlp_dim": 1024, "dim_head": 64},
    )
    pred3 = ConditionedSingleStepPredictor(
        embed_dim=embed_dim,
        macro_action_dim=macro_action_dim,
        **pred_l3_cfg,
    )

    # -- Inverse Dynamics Models --------------------------------------------
    # Each ID model takes (z_current, z_target) -> macro_action_vector.
    id2 = InverseDynamicsModel(embed_dim=embed_dim, macro_action_dim=macro_action_dim)
    id3 = InverseDynamicsModel(embed_dim=embed_dim, macro_action_dim=macro_action_dim)

    # -- Shared components (action encoder + projection heads) --------------
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    # -- Assemble HiJEPA ----------------------------------------------------
    world_model = HiJEPA(
        encoder=encoder,
        pred1=pred1,
        pred2=pred2,
        pred3=pred3,
        id2=id2,
        id3=id3,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(hi_lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ## training             ##
    ##########################
    run_id  = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
