from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from h_le_wm.baseline.adapter import (
    ARPredictor,
    Embedder,
    MLP,
    ModelObjectCallBack,
    SIGReg,
    get_column_normalizer,
    get_img_preprocessor,
)
from h_le_wm.models.jepa import HiJEPA
from h_le_wm.train.pretrained import (
    load_pretrained_low_level_model,
    resolve_pretrained_checkpoint,
)
from h_le_wm.train.steps import (
    build_macro_action_encoder,
    clone_projection_head,
    hi_lejepa_forward,
    hi_lejepa_forward_p2_frozen,
    is_p2_frozen_optimization_enabled,
)
from h_le_wm.train.waypoint_ops import build_p2_frozen_waypoint_collate


def summarize_params(module: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def log_param_breakdown(model: HiJEPA):
    """Print parameter distribution across core modules."""
    parts = [
        ("state_encoder", model.encoder),
        ("p1_low_predictor", model.low_predictor),
        ("p1_action_encoder", model.action_encoder),
        ("projector", model.projector),
        ("p1_low_pred_proj", model.low_pred_proj),
        ("p2_high_pred_proj", model.high_pred_proj),
        ("p2_high_predictor", model.high_predictor),
        ("p2_latent_action_encoder", model.latent_action_encoder),
        ("p2_macro_to_condition", model.macro_to_condition),
    ]

    total_all, trainable_all = summarize_params(model)
    print("[hi_train] parameter breakdown:")
    for name, module in parts:
        total, trainable = summarize_params(module)
        pct = (100.0 * total / total_all) if total_all > 0 else 0.0
        print(
            f"  - {name:24s} total={total:>12,} trainable={trainable:>12,} "
            f"share={pct:6.2f}%"
        )
    print(
        f"[hi_train] total params: {total_all:,} | trainable params: {trainable_all:,} "
        f"({(100.0 * trainable_all / total_all) if total_all > 0 else 0.0:.2f}% trainable)"
    )


class WeightsCheckpointCallback(Callback):
    """Save full Lightning checkpoints at a fixed epoch interval."""

    def __init__(self, dirpath: Path, filename: str, epoch_interval: int = 1):
        super().__init__()
        if epoch_interval <= 0:
            raise ValueError("epoch_interval must be > 0")
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = int(epoch_interval)

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if not trainer.is_global_zero:
            return

        epoch = int(trainer.current_epoch) + 1
        if epoch % self.epoch_interval != 0 and epoch != int(trainer.max_epochs):
            return

        output_path = self.dirpath / f"{self.filename}_epoch_{epoch}_weights.ckpt"
        trainer.save_checkpoint(str(output_path))
        latest_path = self.dirpath / f"{self.filename}_weights.ckpt"
        trainer.save_checkpoint(str(latest_path))


def validate_high_level_config(cfg):
    """Validate high-level config consistency before model construction."""
    history_size = int(cfg.wm.history_size)
    num_steps = int(cfg.data.dataset.num_steps)
    max_span = int(cfg.wm.high_level.waypoints.max_span)
    max_seq_len = int(cfg.latent_action_encoder.max_seq_len)

    if num_steps <= history_size:
        raise ValueError(
            "data.dataset.num_steps must be > wm.history_size to allow future waypoint transitions."
        )

    max_available_span = min(max_span, num_steps - history_size)
    if max_available_span <= 0:
        raise ValueError(
            "No positive waypoint span available. Increase data.dataset.num_steps or reduce "
            "wm.high_level.waypoints.max_span / wm.history_size."
        )

    if max_seq_len < max_available_span:
        raise ValueError(
            "latent_action_encoder.max_seq_len is too small for waypoint sampling. "
            f"Need max_seq_len >= {max_available_span} (effective max span), "
            f"got {max_seq_len}. "
            "Set latent_action_encoder.max_seq_len to wm.high_level.waypoints.max_span "
            "or larger."
        )


@hydra.main(version_base=None, config_path="../config/train", config_name="hi_lewm")
def run(cfg):
    """Main training entrypoint for high-level predictor training.

    Responsibilities:
        - dataset/transforms setup
        - pretrained low-level checkpoint resolution/loading
        - model assembly (frozen low-level + trainable high-level path)
        - optimizer/scheduler wiring
        - trainer/manager launch

    Notes:
        - By default, encoder + low-level modules are frozen.
        - Default objective emphasizes high-level loss (``beta``) for PushT-focused runs.
    """
    validate_high_level_config(cfg)

    use_p2_frozen_optimization = is_p2_frozen_optimization_enabled(cfg)
    if use_p2_frozen_optimization:
        print("[hi_train] enabling P2 frozen input optimization (waypoint-only pixel preprocessing).")

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    pixel_preprocessor = None
    transforms = []
    if use_p2_frozen_optimization:
        pixel_preprocessor = get_img_preprocessor(
            source="pixels",
            target="pixels",
            img_size=cfg.img_size,
        )
    else:
        transforms.append(get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size))

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms) if transforms else None
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    loader_kwargs = dict(cfg.loader)
    if use_p2_frozen_optimization:
        loader_kwargs["collate_fn"] = build_p2_frozen_waypoint_collate(cfg, pixel_preprocessor)

    train = torch.utils.data.DataLoader(
        train_set, **loader_kwargs, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **loader_kwargs, shuffle=False, drop_last=False
    )

    effective_act_dim = int(cfg.data.dataset.frameskip) * int(cfg.wm.action_dim)

    if bool(cfg.pretrained_low_level.enabled):
        ckpt_path = resolve_pretrained_checkpoint(cfg)
        pretrained = load_pretrained_low_level_model(ckpt_path)
        print(f"[hi_train] loaded pretrained low-level object: {ckpt_path}")

        encoder = pretrained.encoder
        low_predictor = pretrained.predictor
        action_encoder = pretrained.action_encoder
        projector = pretrained.projector
        low_predictor_proj = pretrained.pred_proj
        high_predictor_proj = clone_projection_head(pretrained.pred_proj)
    else:
        encoder = spt.backbone.utils.vit_hf(
            cfg.encoder_scale,
            patch_size=cfg.patch_size,
            image_size=cfg.img_size,
            pretrained=False,
            use_mask_token=False,
        )
        hidden_dim = encoder.config.hidden_size
        embed_dim = int(cfg.wm.get("embed_dim", hidden_dim))
        low_predictor = ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        )
        action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
        projector = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )
        low_predictor_proj = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )
        high_predictor_proj = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

    if hasattr(low_predictor, "pos_embedding"):
        embed_dim = int(low_predictor.pos_embedding.shape[-1])
    else:
        embed_dim = int(cfg.wm.get("embed_dim", 192))

    if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
        hidden_dim = int(encoder.config.hidden_size)
    else:
        hidden_dim = embed_dim

    num_waypoints = int(cfg.wm.high_level.waypoints.num)
    if num_waypoints < 3:
        raise ValueError("wm.high_level.waypoints.num must be >= 3")
    high_num_frames = num_waypoints - 1

    high_pred_cfg = dict(cfg.predictor_high)
    high_predictor = ARPredictor(
        num_frames=high_num_frames,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **high_pred_cfg,
    )

    latent_action_dim = int(cfg.wm.high_level.get("latent_action_dim", embed_dim))
    latent_action_encoder = build_macro_action_encoder(
        cfg,
        input_dim=effective_act_dim,
        latent_dim=latent_action_dim,
    )

    cond_dim = embed_dim
    proj_mode = str(cfg.wm.high_level.get("macro_to_condition_proj", "auto"))
    if proj_mode == "identity":
        if latent_action_dim != cond_dim:
            raise ValueError(
                "macro_to_condition_proj=identity requires "
                "latent_action_dim == wm.embed_dim"
            )
        macro_to_condition = torch.nn.Identity()
    elif proj_mode == "linear":
        macro_to_condition = torch.nn.Linear(latent_action_dim, cond_dim)
    elif proj_mode == "auto":
        macro_to_condition = (
            torch.nn.Identity()
            if latent_action_dim == cond_dim
            else torch.nn.Linear(latent_action_dim, cond_dim)
        )
    else:
        raise ValueError(
            f"Unsupported wm.high_level.macro_to_condition_proj={proj_mode}. "
            "Use one of: auto, identity, linear."
        )

    world_model = HiJEPA(
        encoder=encoder,
        low_predictor=low_predictor,
        action_encoder=action_encoder,
        high_predictor=high_predictor,
        latent_action_encoder=latent_action_encoder,
        macro_to_condition=macro_to_condition,
        projector=projector,
        low_pred_proj=low_predictor_proj,
        high_pred_proj=high_predictor_proj,
    )

    freeze_cfg = cfg.pretrained_low_level.freeze
    freeze_encoder = bool(freeze_cfg.get("encoder", True))
    freeze_low_predictor = bool(freeze_cfg.get("low_level_predictor", True))
    freeze_action_encoder = bool(freeze_cfg.get("low_level_action_encoder", True))
    freeze_projector = bool(freeze_cfg.get("projector", True))
    freeze_low_pred_proj = bool(freeze_cfg.get("low_pred_proj", True))
    freeze_high_pred_proj = bool(freeze_cfg.get("high_pred_proj", False))

    if bool(cfg.pretrained_low_level.enabled):
        world_model.freeze_low_level(
            freeze_encoder=freeze_encoder,
            freeze_low_predictor=freeze_low_predictor,
            freeze_action_encoder=freeze_action_encoder,
            freeze_projector=freeze_projector,
            freeze_low_pred_proj=freeze_low_pred_proj,
            freeze_high_pred_proj=freeze_high_pred_proj,
        )
    else:
        if any(
            (
                freeze_encoder,
                freeze_low_predictor,
                freeze_action_encoder,
                freeze_projector,
                freeze_low_pred_proj,
                freeze_high_pred_proj,
            )
        ):
            warnings.warn(
                "pretrained_low_level.enabled=False, so pretrained freeze settings are ignored. "
                "Low-level modules remain trainable.",
                stacklevel=2,
            )
        world_model.freeze_low_level(
            freeze_encoder=False,
            freeze_low_predictor=False,
            freeze_action_encoder=False,
            freeze_projector=False,
            freeze_low_pred_proj=False,
            freeze_high_pred_proj=False,
        )

    log_param_breakdown(world_model)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    selected_forward = (
        hi_lejepa_forward_p2_frozen if use_p2_frozen_optimization else hi_lejepa_forward
    )

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(selected_forward, cfg=cfg),
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
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=int(cfg.checkpointing.object_dump.epoch_interval),
    )

    callbacks = [object_dump_callback]
    if bool(cfg.checkpointing.weights_dump.enabled):
        callbacks.append(
            WeightsCheckpointCallback(
                dirpath=run_dir,
                filename=cfg.output_model_name,
                epoch_interval=int(cfg.checkpointing.weights_dump.epoch_interval),
            )
        )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
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
