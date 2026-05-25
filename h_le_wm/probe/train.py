from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from h_le_wm.baseline.adapter import get_column_normalizer, get_img_preprocessor
from h_le_wm.probe.model import (
    LatentToPixelDecoder,
    compute_psnr,
    compute_ssim,
    denormalize_imagenet,
    epoch_filename,
    format_metric,
    freeze_module,
    infer_latent_dim,
    load_decoder_state_dict,
    load_hi_checkpoint,
    make_comparison_panel,
    save_panel,
    save_probe_bundle,
)
from h_le_wm.train.waypoint_ops import (
    build_action_chunks_batched,
    build_p2_frozen_waypoint_collate,
)


def validate_probe_config(cfg) -> None:
    mode = str(cfg.probe.mode)
    if mode not in {"true_only", "pred_exposed"}:
        raise ValueError("probe.mode must be one of: true_only, pred_exposed")
    if not str(cfg.probe.checkpoint.path or "").strip():
        raise ValueError("probe.checkpoint.path must be set to a frozen HOPE2 object checkpoint.")
    if int(cfg.data.dataset.num_steps) <= int(cfg.wm.history_size):
        raise ValueError("data.dataset.num_steps must exceed wm.history_size.")
    if int(cfg.wm.high_level.waypoints.num) < 3:
        raise ValueError("wm.high_level.waypoints.num must be >= 3.")
    if float(cfg.probe.loss.pred_weight) < 0.0:
        raise ValueError("probe.loss.pred_weight must be >= 0.")


def build_dataset_and_loaders(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    pixel_preprocessor = get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    transforms = []
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            if col == "action":
                setattr(cfg.wm, "action_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(*transforms) if transforms else None
    collate_fn = build_p2_frozen_waypoint_collate(cfg, pixel_preprocessor)

    rnd_gen = torch.Generator().manual_seed(int(cfg.seed))
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[float(cfg.train_split), 1 - float(cfg.train_split)],
        generator=rnd_gen,
    )
    loader_kwargs = dict(cfg.loader)
    loader_kwargs["collate_fn"] = collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_set,
        **loader_kwargs,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        **loader_kwargs,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


class ProbeBundleCallback(Callback):
    def __init__(self, dirpath: Path, filename: str, epoch_interval: int = 1):
        super().__init__()
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
        self.dirpath.mkdir(parents=True, exist_ok=True)
        epoch_path = self.dirpath / epoch_filename(self.filename, epoch, "probe.pt")
        latest_path = self.dirpath / f"{self.filename}_probe.pt"
        metadata = {
            "epoch": epoch,
            "global_step": int(trainer.global_step),
            "mode": str(pl_module.cfg.probe.mode),
            "hi_checkpoint_path": str(pl_module.cfg.probe.checkpoint.path),
        }
        save_probe_bundle(epoch_path, decoder=pl_module.decoder, cfg=pl_module.cfg, metadata=metadata)
        save_probe_bundle(latest_path, decoder=pl_module.decoder, cfg=pl_module.cfg, metadata=metadata)


class DecoderProbeModule(pl.LightningModule):
    def __init__(self, *, hi_model, decoder: LatentToPixelDecoder, cfg, run_dir: Path):
        super().__init__()
        self.hi_model = hi_model
        self.decoder = decoder
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.mode_name = str(cfg.probe.mode)
        self.pred_weight = float(cfg.probe.loss.pred_weight)
        self.max_visuals = int(cfg.probe.visualization.num_grids)
        self.image_log_interval = max(1, int(cfg.wandb.image_log_interval))
        self._val_visuals = None

        self.save_hyperparameters(
            {
                "probe_mode": self.mode_name,
                "img_size": int(cfg.img_size),
                "checkpoint_path": str(cfg.probe.checkpoint.path),
            }
        )

        freeze_module(self.hi_model)

    def configure_optimizers(self):
        optim_type = str(self.cfg.optimizer.type)
        if optim_type != "AdamW":
            raise ValueError(f"Unsupported optimizer.type={optim_type}. Only AdamW is implemented.")
        optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=float(self.cfg.optimizer.lr),
            weight_decay=float(self.cfg.optimizer.weight_decay),
        )
        return optimizer

    def on_fit_start(self):
        freeze_module(self.hi_model)

    def _compute_prediction_latents(
        self, waypoints: torch.Tensor, z_waypoints: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        starts = waypoints[:, :-1]
        ends = waypoints[:, 1:]
        chunk_actions, chunk_mask = build_action_chunks_batched(actions, starts, ends)
        b, k, l_max, act_dim = chunk_actions.shape
        flat_actions = chunk_actions.reshape(b * k, l_max, act_dim)
        flat_mask = chunk_mask.reshape(b * k, l_max)
        flat_macro = self.hi_model.encode_macro_actions(flat_actions, flat_mask)
        macro_actions = flat_macro.reshape(b, k, -1)
        z_context = z_waypoints[:, :-1]
        return self.hi_model.predict_high(z_context, macro_actions)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        waypoints = batch["waypoints"].to(device=self.device, dtype=torch.long)
        pixels = batch["pixels"].to(device=self.device)
        actions = torch.nan_to_num(batch["action"].to(device=self.device), 0.0)

        with torch.no_grad():
            encoded = self.hi_model.encode({"pixels": pixels}, encode_actions=False)
            z_waypoints = encoded["emb"]
            z_target = z_waypoints[:, 1:]
            z_pred = self._compute_prediction_latents(waypoints, z_waypoints, actions)

        target_pixels = pixels[:, 1:]
        context_pixels = pixels[:, :-1]
        flat_target_pixels = target_pixels.reshape(-1, *target_pixels.shape[2:])
        flat_context_pixels = context_pixels.reshape(-1, *context_pixels.shape[2:])
        flat_z_target = z_target.reshape(-1, z_target.size(-1))
        flat_z_pred = z_pred.reshape(-1, z_pred.size(-1))

        decoded_true = self.decoder(flat_z_target)
        true_loss = F.mse_loss(decoded_true, flat_target_pixels)
        pred_loss = torch.zeros_like(true_loss)
        decoded_pred = None
        if self.mode_name == "pred_exposed":
            decoded_pred = self.decoder(flat_z_pred)
            pred_loss = F.mse_loss(decoded_pred, flat_target_pixels)
        loss = true_loss + self.pred_weight * pred_loss

        pred_true_latent_gap = F.mse_loss(flat_z_pred, flat_z_target)
        decoded_true_vis = denormalize_imagenet(decoded_true)
        target_pixels_vis = denormalize_imagenet(flat_target_pixels)
        true_psnr = compute_psnr(decoded_true_vis, target_pixels_vis)
        true_ssim = compute_ssim(decoded_true_vis, target_pixels_vis)

        log_values = {
            f"{stage}/loss": loss,
            f"{stage}/pixel_mse": true_loss,
            f"{stage}/psnr": true_psnr,
            f"{stage}/ssim": true_ssim,
            f"{stage}/latent_norm_true": flat_z_target.norm(dim=-1).mean(),
            f"{stage}/latent_norm_pred": flat_z_pred.norm(dim=-1).mean(),
            f"{stage}/pred_true_latent_gap": pred_true_latent_gap,
        }
        if self.mode_name == "pred_exposed":
            decoded_pred_vis = denormalize_imagenet(decoded_pred)
            pred_psnr = compute_psnr(decoded_pred_vis, target_pixels_vis)
            pred_ssim = compute_ssim(decoded_pred_vis, target_pixels_vis)
            log_values[f"{stage}/pixel_mse_pred"] = pred_loss
            log_values[f"{stage}/psnr_pred"] = pred_psnr
            log_values[f"{stage}/ssim_pred"] = pred_ssim
        self.log_dict(log_values, on_step=(stage == "train"), on_epoch=True, sync_dist=True)

        if stage == "val" and self._val_visuals is None:
            if decoded_pred is None:
                decoded_pred = self.decoder(flat_z_pred)
            self._val_visuals = {
                "context_pixels": flat_context_pixels[: self.max_visuals].detach().cpu(),
                "target_pixels": flat_target_pixels[: self.max_visuals].detach().cpu(),
                "decoded_true": decoded_true[: self.max_visuals].detach().cpu(),
                "decoded_pred": decoded_pred[: self.max_visuals].detach().cpu(),
                "metrics": {
                    "pixel_mse": format_metric(true_loss),
                    "pred_true_latent_gap": format_metric(pred_true_latent_gap),
                    "psnr": format_metric(true_psnr),
                    "ssim": format_metric(true_ssim),
                },
            }
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def on_validation_epoch_start(self):
        self._val_visuals = None

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self._val_visuals is None or not self.trainer.is_global_zero:
            return
        epoch = int(self.current_epoch) + 1
        if epoch % self.image_log_interval != 0:
            return

        panel = make_comparison_panel(
            context_pixels=self._val_visuals["context_pixels"],
            target_pixels=self._val_visuals["target_pixels"],
            decoded_true=self._val_visuals["decoded_true"],
            decoded_pred=self._val_visuals["decoded_pred"],
            max_items=self.max_visuals,
        )
        panel_path = self.run_dir / "visuals" / epoch_filename(self.cfg.output_model_name, epoch, "panel.png")
        save_panel(panel_path, panel)

        if isinstance(self.logger, WandbLogger):
            import wandb

            self.logger.experiment.log(
                {
                    "val/panel": wandb.Image(str(panel_path)),
                    "val/panel_pixel_mse": self._val_visuals["metrics"]["pixel_mse"],
                    "val/panel_psnr": self._val_visuals["metrics"]["psnr"],
                    "val/panel_ssim": self._val_visuals["metrics"]["ssim"],
                    "val/panel_pred_true_latent_gap": self._val_visuals["metrics"]["pred_true_latent_gap"],
                    "trainer/global_step": int(self.global_step),
                },
                step=int(self.global_step),
            )


@hydra.main(version_base=None, config_path="../config/train", config_name="hi_decoder_probe")
def run(cfg):
    torch.set_float32_matmul_precision("high")
    validate_probe_config(cfg)

    train_loader, val_loader = build_dataset_and_loaders(cfg)

    hi_model = load_hi_checkpoint(cfg.probe.checkpoint.path)
    latent_dim = infer_latent_dim(hi_model)
    decoder_cfg = deepcopy(OmegaConf.to_container(cfg.probe.decoder, resolve=True))
    decoder = LatentToPixelDecoder(
        latent_dim=latent_dim,
        img_size=int(cfg.img_size),
        **decoder_cfg,
    )
    init_ckpt = str(cfg.probe.init_decoder_checkpoint or "").strip()
    if init_ckpt:
        decoder.load_state_dict(load_decoder_state_dict(init_ckpt), strict=True)
        print(f"[hi_train_decoder_probe] initialized decoder from: {init_ckpt}")

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if bool(cfg.wandb.enabled):
        wandb_cfg = OmegaConf.to_container(cfg.wandb.config, resolve=True)
        if wandb_cfg.get("entity") in (None, ""):
            wandb_cfg.pop("entity", None)
        logger = WandbLogger(**wandb_cfg)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    callbacks = [
        ProbeBundleCallback(
            dirpath=run_dir,
            filename=cfg.output_model_name,
            epoch_interval=int(cfg.checkpointing.bundle_epoch_interval),
        ),
        ModelCheckpoint(
            dirpath=run_dir,
            filename=cfg.output_model_name + "_epoch_{epoch}",
            every_n_epochs=1,
            save_top_k=-1,
            save_last=True,
        ),
    ]

    module = DecoderProbeModule(
        hi_model=hi_model,
        decoder=decoder,
        cfg=cfg,
        run_dir=run_dir,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )
    resume_cfg = cfg.checkpointing.get("resume_from")
    ckpt_path = None if resume_cfg in (None, "") else str(resume_cfg).strip()
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    run()
