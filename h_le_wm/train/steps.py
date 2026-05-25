from __future__ import annotations

from copy import deepcopy

import torch

from h_le_wm.models.latent_action import LatentActionEncoder
from h_le_wm.models.vq import VQActionEncoder
from h_le_wm.models.waypoint_sampling import sample_waypoints
from h_le_wm.train.waypoint_ops import (
    build_action_chunks_batched,
    gather_waypoint_embeddings,
)


def is_p2_frozen_optimization_enabled(cfg) -> bool:
    """Return whether P2-frozen optimized input pipeline should be enabled."""
    if bool(cfg.training.get("train_low_level", False)):
        return False
    if float(cfg.loss.sigreg.weight) > 0.0:
        return False
    if not bool(cfg.pretrained_low_level.get("enabled", False)):
        return False

    freeze_cfg = cfg.pretrained_low_level.freeze
    required_frozen = (
        bool(freeze_cfg.get("encoder", True)),
        bool(freeze_cfg.get("low_level_predictor", True)),
        bool(freeze_cfg.get("low_level_action_encoder", True)),
        bool(freeze_cfg.get("projector", True)),
        bool(freeze_cfg.get("low_pred_proj", True)),
    )
    return all(required_frozen)


def encode_macro_actions_with_aux(
    model,
    action_chunks: torch.Tensor,
    action_mask: torch.Tensor,
) -> dict:
    """Return macro-actions plus optional auxiliary losses and metrics."""
    if hasattr(model, "encode_macro_actions_with_info"):
        output = model.encode_macro_actions_with_info(action_chunks, action_mask)
        if "macro_actions" not in output:
            raise ValueError("encode_macro_actions_with_info must return `macro_actions`.")
        return output
    macro_actions = model.encode_macro_actions(action_chunks, action_mask)
    return {"macro_actions": macro_actions}


def add_macro_action_aux_losses(output: dict, macro_output: dict, cfg, *, device, dtype) -> None:
    """Accumulate optional VQ auxiliary losses onto the training output dict."""
    zero = torch.zeros((), device=device, dtype=dtype)
    output["vq_recon_loss"] = macro_output.get("recon_loss", zero)
    output["vq_commitment_loss"] = macro_output.get("commitment_loss", zero)
    output["vq_codebook_loss"] = macro_output.get("codebook_loss", zero)
    output["vq_perplexity"] = macro_output.get("perplexity", zero)
    output["vq_active_codes"] = macro_output.get("active_codes", zero)

    vq_cfg = cfg.loss.get("vq", {})
    recon_weight = float(vq_cfg.get("recon_weight", 0.0))
    commitment_weight = float(vq_cfg.get("commitment_weight", 0.0))
    codebook_weight = float(vq_cfg.get("codebook_weight", 0.0))
    output["vq_loss"] = (
        recon_weight * output["vq_recon_loss"]
        + commitment_weight * output["vq_commitment_loss"]
        + codebook_weight * output["vq_codebook_loss"]
    )


def build_macro_action_encoder(cfg, *, input_dim: int, latent_dim: int) -> torch.nn.Module:
    """Instantiate the configured macro-action encoder backend."""
    encoder_cfg = dict(cfg.latent_action_encoder)
    encoder_type = str(encoder_cfg.pop("type", "continuous")).lower()
    vq_cfg = dict(encoder_cfg.pop("vq", {}))

    if encoder_type == "continuous":
        return LatentActionEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            **encoder_cfg,
        )
    if encoder_type == "vq":
        return VQActionEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            **encoder_cfg,
            **vq_cfg,
        )
    raise ValueError(
        f"Unsupported latent_action_encoder.type={encoder_type}. "
        "Use one of: continuous, vq."
    )


def hi_lejepa_forward(self, batch, stage, cfg):
    """Single train or val step for high-level predictor training."""
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    actions = batch["action"]
    b, t, _a = actions.shape
    device = actions.device

    train_low_level = bool(cfg.training.get("train_low_level", False))
    lambd = float(cfg.loss.sigreg.weight)

    waypoints, gaps = sample_waypoints(
        cfg,
        batch_size=b,
        seq_len=t,
        device=device,
    )

    use_waypoint_fast_path = (not train_low_level) and (lambd <= 0.0)
    if use_waypoint_fast_path:
        output = {}
        emb = None
        z_waypoints = self.model.encode_selected_frames(batch["pixels"], waypoints)
    else:
        output = self.model.encode(batch, encode_actions=train_low_level)
        emb = output["emb"]
        z_waypoints = gather_waypoint_embeddings(emb, waypoints)

    z_context = z_waypoints[:, :-1]
    z_target = z_waypoints[:, 1:]

    starts = waypoints[:, :-1]
    ends = waypoints[:, 1:]
    chunk_actions, chunk_mask = build_action_chunks_batched(actions, starts, ends)
    _, k, l_max, act_dim = chunk_actions.shape
    flat_actions = chunk_actions.reshape(b * k, l_max, act_dim)
    flat_mask = chunk_mask.reshape(b * k, l_max)
    macro_output = encode_macro_actions_with_aux(self.model, flat_actions, flat_mask)
    flat_macro = macro_output["macro_actions"]
    macro_actions = flat_macro.reshape(b, k, -1)

    z_pred = self.model.predict_high(z_context, macro_actions)
    output["l2_pred_loss"] = (z_pred - z_target).pow(2).mean()

    if train_low_level:
        if emb is None:
            raise RuntimeError("emb is required for low-level loss but was not computed")
        ctx_len = int(cfg.wm.history_size)
        n_preds = int(cfg.wm.num_preds)
        act_emb = output["act_emb"]
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds : ctx_len + n_preds]
        pred_emb = self.model.predict_low(ctx_emb, ctx_act)
        output["l1_pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    else:
        output["l1_pred_loss"] = torch.zeros((), device=device, dtype=z_waypoints.dtype)

    if lambd > 0.0:
        if emb is None:
            raise RuntimeError("emb is required for SIGReg but was not computed")
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    else:
        output["sigreg_loss"] = torch.zeros((), device=device, dtype=z_waypoints.dtype)

    alpha = float(cfg.loss.get("alpha", 0.0))
    beta = float(cfg.loss.get("beta", 1.0))
    add_macro_action_aux_losses(
        output,
        macro_output,
        cfg,
        device=device,
        dtype=z_waypoints.dtype,
    )
    output["loss"] = (
        alpha * output["l1_pred_loss"]
        + beta * output["l2_pred_loss"]
        + lambd * output["sigreg_loss"]
        + output["vq_loss"]
    )

    output["waypoint_gap_mean"] = gaps.float().mean()
    output["waypoint_gap_max"] = gaps.float().max()
    output["macro_action_norm"] = macro_actions.norm(dim=-1).mean()

    metric_keys = (
        "loss",
        "l1_pred_loss",
        "l2_pred_loss",
        "sigreg_loss",
        "vq_loss",
        "vq_recon_loss",
        "vq_commitment_loss",
        "vq_codebook_loss",
        "vq_perplexity",
        "vq_active_codes",
        "waypoint_gap_mean",
        "waypoint_gap_max",
        "macro_action_norm",
    )
    metrics = {f"{stage}/{k}": output[k].detach() for k in metric_keys}
    self.log_dict(metrics, on_step=True, sync_dist=True)
    return output


def hi_lejepa_forward_p2_frozen(self, batch, stage, cfg):
    """Train or val step for P2-only runs with frozen low-level modules."""
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    actions = batch["action"]
    b, _t, _a = actions.shape
    device = actions.device

    if "waypoints" not in batch:
        raise RuntimeError("P2-frozen forward requires precomputed `batch['waypoints']`.")
    waypoints = batch["waypoints"].to(device=device, dtype=torch.long)
    if waypoints.ndim != 2 or waypoints.size(0) != b:
        raise ValueError("waypoints must be shape (B, N) and match action batch size.")

    output = self.model.encode({"pixels": batch["pixels"]}, encode_actions=False)
    z_waypoints = output["emb"]
    if z_waypoints.size(1) != waypoints.size(1):
        raise RuntimeError("Waypoint pixel count and waypoint index count do not match.")

    z_context = z_waypoints[:, :-1]
    z_target = z_waypoints[:, 1:]

    starts = waypoints[:, :-1]
    ends = waypoints[:, 1:]
    chunk_actions, chunk_mask = build_action_chunks_batched(actions, starts, ends)
    _, k, l_max, act_dim = chunk_actions.shape
    flat_actions = chunk_actions.reshape(b * k, l_max, act_dim)
    flat_mask = chunk_mask.reshape(b * k, l_max)
    macro_output = encode_macro_actions_with_aux(self.model, flat_actions, flat_mask)
    flat_macro = macro_output["macro_actions"]
    macro_actions = flat_macro.reshape(b, k, -1)

    z_pred = self.model.predict_high(z_context, macro_actions)
    output["l2_pred_loss"] = (z_pred - z_target).pow(2).mean()
    output["l1_pred_loss"] = torch.zeros((), device=device, dtype=z_waypoints.dtype)
    output["sigreg_loss"] = torch.zeros((), device=device, dtype=z_waypoints.dtype)

    alpha = float(cfg.loss.get("alpha", 0.0))
    beta = float(cfg.loss.get("beta", 1.0))
    add_macro_action_aux_losses(
        output,
        macro_output,
        cfg,
        device=device,
        dtype=z_waypoints.dtype,
    )
    output["loss"] = (
        alpha * output["l1_pred_loss"]
        + beta * output["l2_pred_loss"]
        + output["vq_loss"]
    )

    gaps = waypoints[:, 1:] - waypoints[:, :-1]
    output["waypoint_gap_mean"] = gaps.float().mean()
    output["waypoint_gap_max"] = gaps.float().max()
    output["macro_action_norm"] = macro_actions.norm(dim=-1).mean()

    metric_keys = (
        "loss",
        "l1_pred_loss",
        "l2_pred_loss",
        "sigreg_loss",
        "vq_loss",
        "vq_recon_loss",
        "vq_commitment_loss",
        "vq_codebook_loss",
        "vq_perplexity",
        "vq_active_codes",
        "waypoint_gap_mean",
        "waypoint_gap_max",
        "macro_action_norm",
    )
    metrics = {f"{stage}/{k}": output[k].detach() for k in metric_keys}
    self.log_dict(metrics, on_step=True, sync_dist=True)
    return output


def clone_projection_head(module: torch.nn.Module) -> torch.nn.Module:
    """Return a trainable deep copy of a projection head module."""
    return deepcopy(module)
