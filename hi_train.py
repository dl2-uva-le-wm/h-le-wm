from functools import partial
from pathlib import Path
import warnings

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from baseline_adapter import (
    Embedder,
    MLP,
    ModelObjectCallBack,
    SIGReg,
    get_column_normalizer,
    get_img_preprocessor,
)
from hi_jepa import HiJEPA
from hi_module import (
    ARPredictorAnchored,
    ConditionedSingleStepPredictor,
    InverseDynamicsModel,
)


def validate_hierarchy_cfg(
    cfg, *, emit_warnings: bool = False
) -> tuple[int, int | None, int, int]:
    """Validate hierarchy settings and return (num_levels, k1, k2, max_offset)."""
    wm_cfg = cfg.wm
    num_levels = int(wm_cfg.get("num_levels", 3))
    if num_levels not in (2, 3):
        raise ValueError(f"wm.num_levels must be 2 or 3, got {num_levels}")

    if "k2" not in wm_cfg:
        raise ValueError(
            "wm.k2 is required for both wm.num_levels=2 and wm.num_levels=3 "
            "(strict break: 2-level now means L3->L1)."
        )
    k2 = int(wm_cfg.k2)
    if k2 <= 0:
        raise ValueError(f"wm.k2 must be > 0, got {k2}")

    if num_levels == 3:
        if "k1" not in wm_cfg:
            raise ValueError("wm.k1 is required when wm.num_levels=3")
        k1 = int(wm_cfg.k1)
        if k1 <= 0:
            raise ValueError(f"wm.k1 must be > 0, got {k1}")
        if k2 <= k1:
            raise ValueError(f"wm.k2 must be > wm.k1 for 3 levels, got k1={k1}, k2={k2}")
    else:
        k1_cfg = int(wm_cfg.get("k1", 0))
        if emit_warnings and k1_cfg != 0:
            warnings.warn(
                "wm.k1 is ignored when wm.num_levels=2 (L3->L1 topology). "
                "Set wm.k1=0 to silence this warning.",
                stacklevel=2,
            )
        k1 = None

    max_offset = k2
    return num_levels, k1, k2, max_offset


def hi_lejepa_forward(self, batch, stage, cfg):
    """Training/validation step for 2-level or 3-level hierarchical LeWM."""
    num_levels, k1, k2, max_offset = validate_hierarchy_cfg(cfg)

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    alpha = cfg.loss.inverse_dynamics.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    T = emb.size(1)
    t_idx = ctx_len - 1
    if T < ctx_len + max_offset:
        raise ValueError(
            f"Sequence length {T} is too short for ctx_len={ctx_len} + max_offset={max_offset}. "
            f"Set data.sequence_length >= {ctx_len + max_offset}."
        )

    z_t = emb[:, t_idx]
    z_tk2 = emb[:, t_idx + k2]

    if self.model.id3 is None or self.model.pred3 is None:
        raise RuntimeError("Both num_levels=2 and num_levels=3 require id3/pred3 on model")

    a3 = self.model.id3(z_t, z_tk2)
    z3_pred = self.model.pred3(z_t, a3)
    output["l3_pred_loss"] = (z3_pred - z_tk2).pow(2).mean()

    if num_levels == 3:
        assert k1 is not None
        if self.model.id2 is None or self.model.pred2 is None:
            raise RuntimeError("num_levels=3 requires id2/pred2 on model")

        z_tk1 = emb[:, t_idx + k1]
        a2 = self.model.id2(z_t, z3_pred.detach())
        z2_pred = self.model.pred2(z_t, a2, z_anchor=z3_pred.detach())
        output["l2_pred_loss"] = (z2_pred - z_tk1).pow(2).mean()
        output["act_reg_loss"] = a2.pow(2).mean() + a3.pow(2).mean()
        l1_anchor = z2_pred.detach()
    else:
        output["act_reg_loss"] = a3.pow(2).mean()
        l1_anchor = z3_pred.detach()

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_l1 = emb[:, n_preds : ctx_len + n_preds]
    pred_l1 = self.model.predict(ctx_emb, ctx_act, z_anchor=l1_anchor)
    output["l1_pred_loss"] = (pred_l1 - tgt_l1).pow(2).mean()

    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    output["loss"] = (
        output["l1_pred_loss"]
        + output["l3_pred_loss"]
        + lambd * output["sigreg_loss"]
        + alpha * output["act_reg_loss"]
    )
    if num_levels == 3:
        output["loss"] = output["loss"] + output["l2_pred_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def summarize_params(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


@hydra.main(version_base=None, config_path="./config/train", config_name="hi_lewm")
def run(cfg):
    num_levels, _k1, _k2, _max_offset = validate_hierarchy_cfg(cfg, emit_warnings=True)

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

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    macro_action_dim = cfg.wm.get("macro_action_dim", embed_dim)

    pred1 = ARPredictorAnchored(
        embed_dim=embed_dim,
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    pred2 = None
    id2 = None
    if num_levels == 3:
        pred_l2_cfg = cfg.get(
            "predictor_l2",
            {"depth": 3, "heads": 8, "mlp_dim": 1024, "dim_head": 64},
        )
        pred2 = ConditionedSingleStepPredictor(
            embed_dim=embed_dim,
            macro_action_dim=macro_action_dim,
            **pred_l2_cfg,
        )
        id2 = InverseDynamicsModel(embed_dim=embed_dim, macro_action_dim=macro_action_dim)

    pred_l3_cfg = cfg.get(
        "predictor_l3",
        {"depth": 3, "heads": 8, "mlp_dim": 1024, "dim_head": 64},
    )
    pred3 = ConditionedSingleStepPredictor(
        embed_dim=embed_dim,
        macro_action_dim=macro_action_dim,
        **pred_l3_cfg,
    )
    id3 = InverseDynamicsModel(embed_dim=embed_dim, macro_action_dim=macro_action_dim)

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

    world_model = HiJEPA(
        encoder=encoder,
        pred1=pred1,
        action_encoder=action_encoder,
        pred2=pred2,
        pred3=pred3,
        id2=id2,
        id3=id3,
        num_levels=num_levels,
        projector=projector,
        pred_proj=predictor_proj,
    )
    total_params, trainable_params = summarize_params(world_model)
    print(f"[hi_train] model params total={total_params:,} trainable={trainable_params:,}")

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

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        wandb_cfg = OmegaConf.to_container(cfg.wandb.config, resolve=True)
        if wandb_cfg.get("entity") in (None, ""):
            wandb_cfg.pop("entity", None)

        logger = WandbLogger(**wandb_cfg)
        try:
            logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        except Exception as exc:
            # Common cluster failure mode: an explicit but invalid entity slug.
            if (
                wandb_cfg.get("entity")
                and "entity" in str(exc).lower()
                and "not found" in str(exc).lower()
            ):
                bad_entity = wandb_cfg["entity"]
                warnings.warn(
                    f"W&B entity '{bad_entity}' not found; retrying with default logged-in entity.",
                    stacklevel=2,
                )
                wandb_cfg.pop("entity", None)
                logger = WandbLogger(**wandb_cfg)
                logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
            else:
                raise

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


if __name__ == "__main__":
    run()
