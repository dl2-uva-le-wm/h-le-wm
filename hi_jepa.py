from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class HiJEPA(nn.Module):
    """Two-level JEPA model focused on high-level predictor training."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        low_predictor: nn.Module,
        action_encoder: nn.Module,
        high_predictor: nn.Module,
        latent_action_encoder: nn.Module,
        macro_to_condition: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.low_predictor = low_predictor
        self.action_encoder = action_encoder
        self.high_predictor = high_predictor
        self.latent_action_encoder = latent_action_encoder
        self.macro_to_condition = macro_to_condition
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self._freeze_flags = {
            "encoder": False,
            "low_predictor": False,
            "action_encoder": False,
            "projector": False,
            "pred_proj": False,
        }

    def _set_requires_grad(self, module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad_(requires_grad)

    def _enforce_frozen_eval_mode(self):
        """Keep frozen low-level modules in eval mode during training."""
        if self._freeze_flags["encoder"]:
            self.encoder.eval()
        if self._freeze_flags["low_predictor"]:
            self.low_predictor.eval()
        if self._freeze_flags["action_encoder"]:
            self.action_encoder.eval()
        if self._freeze_flags["projector"]:
            self.projector.eval()
        if self._freeze_flags["pred_proj"]:
            self.pred_proj.eval()

    def freeze_low_level(
        self,
        *,
        freeze_encoder: bool = True,
        freeze_low_predictor: bool = True,
        freeze_action_encoder: bool = True,
        freeze_projector: bool = True,
        freeze_pred_proj: bool = True,
    ):
        self._freeze_flags["encoder"] = bool(freeze_encoder)
        self._freeze_flags["low_predictor"] = bool(freeze_low_predictor)
        self._freeze_flags["action_encoder"] = bool(freeze_action_encoder)
        self._freeze_flags["projector"] = bool(freeze_projector)
        self._freeze_flags["pred_proj"] = bool(freeze_pred_proj)

        self._set_requires_grad(self.encoder, not self._freeze_flags["encoder"])
        self._set_requires_grad(self.low_predictor, not self._freeze_flags["low_predictor"])
        self._set_requires_grad(self.action_encoder, not self._freeze_flags["action_encoder"])
        self._set_requires_grad(self.projector, not self._freeze_flags["projector"])
        self._set_requires_grad(self.pred_proj, not self._freeze_flags["pred_proj"])
        self._enforce_frozen_eval_mode()

    def train(self, mode: bool = True):
        super().train(mode)
        self._enforce_frozen_eval_mode()
        return self

    def encode(self, info: dict, *, encode_actions: bool = True) -> dict:
        """Encode observation sequence (and optionally low-level action sequence).

        Args:
            info: Batch dict containing at least ``pixels`` and optionally ``action``.
            encode_actions: If True and ``action`` exists, compute ``act_emb``.

        Returns:
            ``info`` augmented with:
                - ``emb``: (B, T, D_z)
                - ``act_emb``: (B, T, D_z), only when requested
        """
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if encode_actions and "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict_low(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        # emb: (B, T_ctx, D_z), act_emb: (B, T_ctx, D_z)
        preds = self.low_predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def encode_macro_actions(
        self,
        action_chunks: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # action_chunks: (B, T_chunk, A), action_mask: (B, T_chunk)
        return self.latent_action_encoder(action_chunks, action_mask=action_mask)

    def project_macro_to_condition_space(self, macro_actions: torch.Tensor) -> torch.Tensor:
        if macro_actions.ndim != 3:
            raise ValueError("macro_actions must be shape (B, T, latent_action_dim)")
        # macro_actions: (B, T_wp, D_l) -> conditioning: (B, T_wp, D_z)
        projected = self.macro_to_condition(macro_actions)
        return projected

    def predict_high(self, emb: torch.Tensor, macro_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb: (B, T, D) waypoint context latents
            macro_actions: (B, T, latent_action_dim)
        Returns:
            predicted waypoint latents: (B, T, D)
        """
        # emb: (B, T_wp, D_z), macro_actions: (B, T_wp, D_l)
        high_cond = self.project_macro_to_condition_space(macro_actions)  # (B, T_wp, D_z)
        preds = self.high_predictor(emb, high_cond)  # (B, T_wp, D_h)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds
