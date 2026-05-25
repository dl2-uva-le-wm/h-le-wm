from __future__ import annotations

from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from h_le_wm.probe.model import (
    LatentToPixelDecoder,
    compute_psnr,
    compute_ssim,
    denormalize_imagenet,
    epoch_filename,
    format_metric,
    infer_latent_dim,
    load_decoder_state_dict,
    load_hi_checkpoint,
    make_comparison_panel,
    save_panel,
)
from h_le_wm.train.waypoint_ops import build_action_chunks_batched
from h_le_wm.probe.train import build_dataset_and_loaders, validate_probe_config


@torch.inference_mode()
def evaluate_loader(cfg, hi_model, decoder, loader, output_dir: Path) -> dict[str, float]:
    hi_model.eval()
    decoder.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    sum_true_mse = 0.0
    sum_pred_mse = 0.0
    sum_latent_gap = 0.0
    sum_psnr_true = 0.0
    sum_psnr_pred = 0.0
    sum_ssim_true = 0.0
    sum_ssim_pred = 0.0
    num_batches = 0
    num_panels = 0

    for batch_idx, batch in enumerate(loader):
        if cfg.eval.max_batches not in (None, "") and batch_idx >= int(cfg.eval.max_batches):
            break

        waypoints = batch["waypoints"].to(dtype=torch.long)
        pixels = batch["pixels"]
        actions = torch.nan_to_num(batch["action"], 0.0)
        device = next(decoder.parameters()).device
        waypoints = waypoints.to(device)
        pixels = pixels.to(device)
        actions = actions.to(device)

        encoded = hi_model.encode({"pixels": pixels}, encode_actions=False)
        z_waypoints = encoded["emb"]
        z_target = z_waypoints[:, 1:]

        starts = waypoints[:, :-1]
        ends = waypoints[:, 1:]
        chunk_actions, chunk_mask = build_action_chunks_batched(actions, starts, ends)
        b, k, l_max, act_dim = chunk_actions.shape
        flat_actions = chunk_actions.reshape(b * k, l_max, act_dim)
        flat_mask = chunk_mask.reshape(b * k, l_max)
        flat_macro = hi_model.encode_macro_actions(flat_actions, flat_mask)
        z_pred = hi_model.predict_high(z_waypoints[:, :-1], flat_macro.reshape(b, k, -1))

        flat_context_pixels = pixels[:, :-1].reshape(-1, *pixels.shape[2:])
        flat_target_pixels = pixels[:, 1:].reshape(-1, *pixels.shape[2:])
        flat_z_target = z_target.reshape(-1, z_target.size(-1))
        flat_z_pred = z_pred.reshape(-1, z_pred.size(-1))

        decoded_true = decoder(flat_z_target)
        decoded_pred = decoder(flat_z_pred)

        true_mse = F.mse_loss(decoded_true, flat_target_pixels)
        pred_mse = F.mse_loss(decoded_pred, flat_target_pixels)
        latent_gap = F.mse_loss(flat_z_pred, flat_z_target)
        decoded_true_vis = denormalize_imagenet(decoded_true)
        decoded_pred_vis = denormalize_imagenet(decoded_pred)
        target_pixels_vis = denormalize_imagenet(flat_target_pixels)
        psnr_true = compute_psnr(decoded_true_vis, target_pixels_vis)
        psnr_pred = compute_psnr(decoded_pred_vis, target_pixels_vis)
        ssim_true = compute_ssim(decoded_true_vis, target_pixels_vis)
        ssim_pred = compute_ssim(decoded_pred_vis, target_pixels_vis)

        sum_true_mse += format_metric(true_mse)
        sum_pred_mse += format_metric(pred_mse)
        sum_latent_gap += format_metric(latent_gap)
        sum_psnr_true += format_metric(psnr_true)
        sum_psnr_pred += format_metric(psnr_pred)
        sum_ssim_true += format_metric(ssim_true)
        sum_ssim_pred += format_metric(ssim_pred)
        num_batches += 1

        if num_panels < int(cfg.probe.visualization.num_grids):
            panel = make_comparison_panel(
                context_pixels=flat_context_pixels.detach().cpu(),
                target_pixels=flat_target_pixels.detach().cpu(),
                decoded_true=decoded_true.detach().cpu(),
                decoded_pred=decoded_pred.detach().cpu(),
                max_items=min(4, flat_target_pixels.size(0)),
            )
            panel_path = output_dir / epoch_filename("batch", batch_idx, "panel.png")
            save_panel(panel_path, panel)
            num_panels += 1

    if num_batches == 0:
        raise RuntimeError("No evaluation batches were processed.")

    return {
        "pixel_mse_true": sum_true_mse / num_batches,
        "pixel_mse_pred": sum_pred_mse / num_batches,
        "pred_true_latent_gap": sum_latent_gap / num_batches,
        "psnr_true": sum_psnr_true / num_batches,
        "psnr_pred": sum_psnr_pred / num_batches,
        "ssim_true": sum_ssim_true / num_batches,
        "ssim_pred": sum_ssim_pred / num_batches,
    }


@hydra.main(version_base=None, config_path="../config/train", config_name="hi_decoder_probe")
def run(cfg):
    validate_probe_config(cfg)
    decoder_ckpt = str(cfg.eval.decoder_checkpoint_path or "").strip()
    if not decoder_ckpt:
        raise ValueError("eval.decoder_checkpoint_path must be set.")

    train_loader, val_loader = build_dataset_and_loaders(cfg)
    loader = train_loader if str(cfg.eval.split) == "train" else val_loader

    hi_model = load_hi_checkpoint(cfg.probe.checkpoint.path)
    latent_dim = infer_latent_dim(hi_model)
    decoder_cfg = OmegaConf.to_container(cfg.probe.decoder, resolve=True)
    decoder = LatentToPixelDecoder(latent_dim=latent_dim, img_size=int(cfg.img_size), **decoder_cfg)
    decoder.load_state_dict(load_decoder_state_dict(decoder_ckpt), strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hi_model = hi_model.to(device)
    decoder = decoder.to(device)

    base_dir = Path(swm.data.utils.get_cache_dir(), str(cfg.subdir or ""))
    output_dir = base_dir / str(cfg.eval.output_subdir)
    metrics = evaluate_loader(cfg, hi_model, decoder, loader, output_dir)

    print("[hi_decoder_probe_eval] metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.6f}")


if __name__ == "__main__":
    run()
