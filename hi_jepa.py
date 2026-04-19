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
        low_pred_proj: nn.Module | None = None,
        high_pred_proj: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.low_predictor = low_predictor
        self.action_encoder = action_encoder
        self.high_predictor = high_predictor
        self.latent_action_encoder = latent_action_encoder
        self.macro_to_condition = macro_to_condition
        self.projector = projector or nn.Identity()
        self.low_pred_proj = low_pred_proj or nn.Identity()
        self.high_pred_proj = high_pred_proj or nn.Identity()
        self._freeze_flags = {
            "encoder": False,
            "low_predictor": False,
            "action_encoder": False,
            "projector": False,
            "low_pred_proj": False,
            "high_pred_proj": False,
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
        if self._freeze_flags["low_pred_proj"]:
            self.low_pred_proj.eval()
        if self._freeze_flags["high_pred_proj"]:
            self.high_pred_proj.eval()

    def freeze_low_level(
        self,
        *,
        freeze_encoder: bool = True,
        freeze_low_predictor: bool = True,
        freeze_action_encoder: bool = True,
        freeze_projector: bool = True,
        freeze_low_pred_proj: bool = True,
        freeze_high_pred_proj: bool = False,
    ):
        self._freeze_flags["encoder"] = bool(freeze_encoder)
        self._freeze_flags["low_predictor"] = bool(freeze_low_predictor)
        self._freeze_flags["action_encoder"] = bool(freeze_action_encoder)
        self._freeze_flags["projector"] = bool(freeze_projector)
        self._freeze_flags["low_pred_proj"] = bool(freeze_low_pred_proj)
        self._freeze_flags["high_pred_proj"] = bool(freeze_high_pred_proj)

        self._set_requires_grad(self.encoder, not self._freeze_flags["encoder"])
        self._set_requires_grad(self.low_predictor, not self._freeze_flags["low_predictor"])
        self._set_requires_grad(self.action_encoder, not self._freeze_flags["action_encoder"])
        self._set_requires_grad(self.projector, not self._freeze_flags["projector"])
        self._set_requires_grad(self.low_pred_proj, not self._freeze_flags["low_pred_proj"])
        self._set_requires_grad(self.high_pred_proj, not self._freeze_flags["high_pred_proj"])
        self._enforce_frozen_eval_mode()

    def train(self, mode: bool = True):
        super().train(mode)
        self._enforce_frozen_eval_mode()
        return self

    def _encode_pixels_sequence(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode a pixel sequence ``(B, T, C, H, W)`` into latents ``(B, T, D_z)``."""
        if pixels.ndim != 5:
            raise ValueError("pixels must be shape (B, T, C, H, W)")
        b = pixels.size(0)
        flat_pixels = rearrange(pixels.float(), "b t ... -> (b t) ...")
        output = self.encoder(flat_pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        return rearrange(emb, "(b t) d -> b t d", b=b)

    def encode_selected_frames(
        self,
        pixels: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Encode selected frame indices from a sequence.

        Args:
            pixels: Full pixel sequence ``(B, T, C, H, W)``.
            frame_indices: Per-sample frame indices ``(B, N)``.

        Returns:
            Selected-frame latents ``(B, N, D_z)``.
        """
        if pixels.ndim != 5:
            raise ValueError("pixels must be shape (B, T, C, H, W)")
        if frame_indices.ndim != 2:
            raise ValueError("frame_indices must be shape (B, N)")

        b, t = pixels.shape[:2]
        if frame_indices.size(0) != b:
            raise ValueError("frame_indices batch dimension must match pixels")

        indices = frame_indices.to(device=pixels.device, dtype=torch.long)
        if (indices < 0).any() or (indices >= t).any():
            raise ValueError("frame_indices must be in [0, T)")

        batch_idx = torch.arange(b, device=pixels.device).unsqueeze(1)
        selected_pixels = pixels[batch_idx, indices]  # (B, N, C, H, W)
        return self._encode_pixels_sequence(selected_pixels)

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
        info["emb"] = self._encode_pixels_sequence(info["pixels"])

        if encode_actions and "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict_low(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        # emb: (B, T_ctx, D_z), act_emb: (B, T_ctx, D_z)
        preds = self.low_predictor(emb, act_emb)
        preds = self.low_pred_proj(rearrange(preds, "b t d -> (b t) d"))
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
        preds = self.high_pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    # -------------------------------------------------------------------------
    # Inference / planning API (compatible with stable_worldmodel AutoCostModel)
    # -------------------------------------------------------------------------
    def _device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _infer_low_context_len(self) -> int:
        if hasattr(self.low_predictor, "pos_embedding"):
            return int(self.low_predictor.pos_embedding.shape[1])
        return 1

    def _infer_high_context_len(self) -> int:
        if hasattr(self.high_predictor, "pos_embedding"):
            return int(self.high_predictor.pos_embedding.shape[1])
        return 1

    def _infer_latent_action_dim(self) -> int:
        if hasattr(self.latent_action_encoder, "output_proj"):
            return int(self.latent_action_encoder.output_proj.out_features)
        if hasattr(self.latent_action_encoder, "latent_dim"):
            return int(self.latent_action_encoder.latent_dim)
        raise ValueError("Unable to infer latent action dimension from latent_action_encoder")

    def _expand_sample_dim(self, x: torch.Tensor, *, target_samples: int) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(1).expand(-1, target_samples, -1)
        if x.ndim == 3:
            if x.shape[1] == target_samples:
                return x
            if x.shape[1] == 1:
                return x.expand(-1, target_samples, -1)
            raise ValueError(
                f"Unexpected sample dimension for tensor {tuple(x.shape)} "
                f"(expected 1 or {target_samples})"
            )
        if x.ndim == 4:
            if x.shape[1] == target_samples:
                return x
            if x.shape[1] == 1:
                return x.expand(-1, target_samples, -1, -1)
            raise ValueError(
                f"Unexpected sample dimension for tensor {tuple(x.shape)} "
                f"(expected 1 or {target_samples})"
            )
        raise ValueError(f"Unsupported tensor rank for sample expansion: {x.ndim}")

    def _encode_goal_from_info(self, info_dict: dict, num_samples: int, device: torch.device) -> torch.Tensor:
        if "goal" not in info_dict:
            raise ValueError("info_dict must contain 'goal' for flat planning cost.")
        goal = info_dict["goal"]
        if not torch.is_tensor(goal):
            raise ValueError("info_dict['goal'] must be a torch.Tensor")
        goal = goal.to(device)
        if goal.ndim == 5:
            # (B, T, C, H, W) -> (B, 1, D)
            b = goal.size(0)
            goal_batch = {"pixels": goal}
            goal_out = self.encode(goal_batch, encode_actions=False)
            z_goal = goal_out["emb"][:, -1].unsqueeze(1)
            return z_goal.expand(b, num_samples, -1)
        if goal.ndim == 6:
            # (B, S, T, C, H, W) -> (B, S, D)
            b, s = goal.shape[:2]
            flat_goal = goal.reshape(b * s, *goal.shape[2:])
            goal_batch = {"pixels": flat_goal}
            goal_out = self.encode(goal_batch, encode_actions=False)
            z_goal = goal_out["emb"][:, -1].reshape(b, s, -1)
            return self._expand_sample_dim(z_goal, target_samples=num_samples)
        raise ValueError(f"Unsupported goal tensor shape: {tuple(goal.shape)}")

    def _encode_init_from_info(self, info_dict: dict, num_samples: int, device: torch.device) -> torch.Tensor:
        if "pixels" not in info_dict:
            raise ValueError("info_dict must contain 'pixels' for flat planning cost.")
        pixels = info_dict["pixels"]
        if not torch.is_tensor(pixels):
            raise ValueError("info_dict['pixels'] must be a torch.Tensor")
        pixels = pixels.to(device)
        if pixels.ndim == 5:
            # (B, T, C, H, W) -> (B, 1, D)
            b = pixels.size(0)
            batch = {"pixels": pixels}
            out = self.encode(batch, encode_actions=False)
            z_init = out["emb"][:, -1].unsqueeze(1)
            return z_init.expand(b, num_samples, -1)
        if pixels.ndim == 6:
            # (B, S, T, C, H, W) -> (B, S, D)
            b, s = pixels.shape[:2]
            flat_pixels = pixels.reshape(b * s, *pixels.shape[2:])
            batch = {"pixels": flat_pixels}
            out = self.encode(batch, encode_actions=False)
            z_init = out["emb"][:, -1].reshape(b, s, -1)
            return self._expand_sample_dim(z_init, target_samples=num_samples)
        raise ValueError(f"Unsupported pixels tensor shape: {tuple(pixels.shape)}")

    @torch.inference_mode()
    def rollout_high(self, z_init: torch.Tensor, latent_actions: torch.Tensor) -> torch.Tensor:
        """Autoregressive high-level rollout.

        Args:
            z_init: (B, D) or (B, S, D)
            latent_actions: (B, S, T, D_l) or (B, T, D_l)
        Returns:
            Predicted high-level latents (B, S, T, D)
        """
        if latent_actions.ndim == 3:
            latent_actions = latent_actions.unsqueeze(1)
        if latent_actions.ndim != 4:
            raise ValueError(
                "latent_actions must have shape (B, S, T, D_l) or (B, T, D_l)"
            )

        device = self._device()
        latent_actions = latent_actions.to(device)
        b, s, t, d_l = latent_actions.shape

        if z_init.ndim == 2:
            z_init = z_init.unsqueeze(1).expand(-1, s, -1)
        elif z_init.ndim == 3:
            z_init = self._expand_sample_dim(z_init, target_samples=s)
        else:
            raise ValueError("z_init must have shape (B, D) or (B, S, D)")
        z_init = z_init.to(device)

        bs = b * s
        z_hist = z_init.reshape(bs, 1, -1)
        act_hist = []
        preds = []
        high_ctx = self._infer_high_context_len()

        for k in range(t):
            act_k = latent_actions[:, :, k, :].reshape(bs, d_l)
            act_hist.append(act_k)
            act_seq = torch.stack(act_hist, dim=1)
            ctx = min(high_ctx, z_hist.size(1), act_seq.size(1))
            z_ctx = z_hist[:, -ctx:]
            a_ctx = act_seq[:, -ctx:]
            z_next = self.predict_high(z_ctx, a_ctx)[:, -1:, :]
            z_hist = torch.cat([z_hist, z_next], dim=1)
            preds.append(z_next.squeeze(1))

        pred_seq = torch.stack(preds, dim=1).reshape(b, s, t, -1)
        return pred_seq

    @torch.inference_mode()
    def rollout_low(
        self,
        z_hist: torch.Tensor,
        a_hist: torch.Tensor | None,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive low-level rollout.

        Args:
            z_hist: (B, H, D) or (B, S, H, D) or (B, S, D)
            a_hist: (B, H, A) or (B, S, H, A) or (B, S, A) or None
            future_actions: (B, S, T, A) or (B, T, A)
        Returns:
            Predicted low-level latents (B, S, T, D)
        """
        if future_actions.ndim == 3:
            future_actions = future_actions.unsqueeze(1)
        if future_actions.ndim != 4:
            raise ValueError(
                "future_actions must have shape (B, S, T, A) or (B, T, A)"
            )

        device = self._device()
        future_actions = future_actions.to(device)
        b, s, t, a_dim = future_actions.shape

        # Normalize z_hist to (B, S, H, D)
        if z_hist.ndim == 2:
            z_hist = z_hist.unsqueeze(1).unsqueeze(2).expand(-1, s, -1, -1)
        elif z_hist.ndim == 3:
            if z_hist.shape[1] == s:
                z_hist = z_hist.unsqueeze(2)
            else:
                z_hist = z_hist.unsqueeze(1).expand(-1, s, -1, -1)
        elif z_hist.ndim == 4:
            z_hist = self._expand_sample_dim(z_hist, target_samples=s)
        else:
            raise ValueError("z_hist must have shape (B, H, D), (B, S, D), or (B, S, H, D)")
        z_hist = z_hist.to(device)

        # Normalize a_hist to (B, S, H, A)
        if a_hist is None:
            a_hist = torch.zeros(
                (b, s, z_hist.size(2), a_dim),
                device=device,
                dtype=future_actions.dtype,
            )
        elif not torch.is_tensor(a_hist):
            raise ValueError("a_hist must be a torch.Tensor or None")
        else:
            a_hist = a_hist.to(device)
            if a_hist.ndim == 2:
                a_hist = a_hist.unsqueeze(1).unsqueeze(2).expand(-1, s, -1, -1)
            elif a_hist.ndim == 3:
                if a_hist.shape[1] == s:
                    a_hist = a_hist.unsqueeze(2)
                else:
                    a_hist = a_hist.unsqueeze(1).expand(-1, s, -1, -1)
            elif a_hist.ndim == 4:
                a_hist = self._expand_sample_dim(a_hist, target_samples=s)
            else:
                raise ValueError(
                    "a_hist must have shape (B, H, A), (B, S, A), or (B, S, H, A)"
                )

        bs = b * s
        z_seq = z_hist.reshape(bs, z_hist.size(2), z_hist.size(3))
        a_seq = a_hist.reshape(bs, a_hist.size(2), a_hist.size(3))
        preds = []
        low_ctx = self._infer_low_context_len()

        for k in range(t):
            a_k = future_actions[:, :, k, :].reshape(bs, 1, a_dim)
            a_seq = torch.cat([a_seq, a_k], dim=1)
            act_emb = self.action_encoder(a_seq)
            ctx = min(low_ctx, z_seq.size(1), act_emb.size(1))
            z_ctx = z_seq[:, -ctx:]
            a_ctx = act_emb[:, -ctx:]
            z_next = self.predict_low(z_ctx, a_ctx)[:, -1:, :]
            z_seq = torch.cat([z_seq, z_next], dim=1)
            preds.append(z_next.squeeze(1))

        pred_seq = torch.stack(preds, dim=1).reshape(b, s, t, -1)
        return pred_seq

    @torch.inference_mode()
    def get_cost_high(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """High-level latent-action planning cost E2."""
        if not torch.is_tensor(action_candidates):
            raise ValueError("action_candidates must be a torch.Tensor")
        if action_candidates.ndim != 4:
            raise ValueError(
                "action_candidates must have shape (B, S, H, D_high)"
            )

        device = self._device()
        action_candidates = action_candidates.to(device)
        b, s, h, d_high = action_candidates.shape

        z_init = info_dict.get("z_init")
        z_goal = info_dict.get("z_goal")
        if not torch.is_tensor(z_init) or not torch.is_tensor(z_goal):
            raise ValueError("High-level cost requires tensor keys: z_init and z_goal")

        z_init = self._expand_sample_dim(z_init.to(device), target_samples=s)
        z_goal = self._expand_sample_dim(z_goal.to(device), target_samples=s)

        latent_dim = self._infer_latent_action_dim()
        if d_high % latent_dim != 0:
            raise ValueError(
                f"High-level candidate dim ({d_high}) is not divisible by latent_action_dim ({latent_dim})."
            )
        high_action_block = d_high // latent_dim
        latent_actions = action_candidates.reshape(b, s, h * high_action_block, latent_dim)

        pred_rollout = self.rollout_high(z_init, latent_actions)
        z_final = pred_rollout[:, :, -1, :]
        cost = (z_final - z_goal).pow(2).sum(dim=-1)
        return cost

    @torch.inference_mode()
    def get_cost_low(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Low-level primitive-action planning cost E1."""
        if not torch.is_tensor(action_candidates):
            raise ValueError("action_candidates must be a torch.Tensor")
        if action_candidates.ndim != 4:
            raise ValueError(
                "action_candidates must have shape (B, S, H, D_low)"
            )

        device = self._device()
        action_candidates = action_candidates.to(device)
        b, s, _h, _d = action_candidates.shape

        z_hist = info_dict.get("z_hist")
        a_hist = info_dict.get("a_hist")
        z_subgoal = info_dict.get("z_subgoal")

        if not torch.is_tensor(z_hist):
            raise ValueError("Low-level cost requires tensor key: z_hist")
        if not torch.is_tensor(z_subgoal):
            raise ValueError("Low-level cost requires tensor key: z_subgoal")

        z_hist = z_hist.to(device)
        z_subgoal = self._expand_sample_dim(z_subgoal.to(device), target_samples=s)
        a_hist_t = a_hist.to(device) if torch.is_tensor(a_hist) else None

        pred_rollout = self.rollout_low(z_hist, a_hist_t, action_candidates)
        z_final = pred_rollout[:, :, -1, :]
        cost = (z_final - z_subgoal).pow(2).sum(dim=-1)
        return cost

    @torch.inference_mode()
    def get_cost_flat(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Flat low-level planning fallback for non-hierarchical evaluation."""
        if not torch.is_tensor(action_candidates):
            raise ValueError("action_candidates must be a torch.Tensor")
        if action_candidates.ndim != 4:
            raise ValueError(
                "action_candidates must have shape (B, S, H, D_low)"
            )

        device = self._device()
        action_candidates = action_candidates.to(device)
        _b, s, _h, _d = action_candidates.shape

        z_init = self._encode_init_from_info(info_dict, s, device)
        z_goal = self._encode_goal_from_info(info_dict, s, device)
        z_hist = z_init.unsqueeze(2)  # (B, S, 1, D)
        pred_rollout = self.rollout_low(z_hist, None, action_candidates)
        z_final = pred_rollout[:, :, -1, :]
        cost = (z_final - z_goal).pow(2).sum(dim=-1)
        return cost

    @torch.inference_mode()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Planning entrypoint used by stable_worldmodel solvers.

        Supported modes:
            - planner_level='high' -> high-level latent planning
            - planner_level='low'  -> low-level primitive planning
            - missing planner_level -> inferred from tensor keys, then flat fallback
        """
        planner_level = info_dict.get("planner_level", None)
        if planner_level == "high":
            return self.get_cost_high(info_dict, action_candidates)
        if planner_level == "low":
            return self.get_cost_low(info_dict, action_candidates)

        # Some solver paths may drop non-tensor metadata keys (e.g. planner_level).
        # Infer routing from the latent tensors that the corresponding planners pass.
        has_high_latents = torch.is_tensor(info_dict.get("z_init")) and torch.is_tensor(info_dict.get("z_goal"))
        if has_high_latents:
            return self.get_cost_high(info_dict, action_candidates)

        has_low_latents = torch.is_tensor(info_dict.get("z_hist")) and torch.is_tensor(info_dict.get("z_subgoal"))
        if has_low_latents:
            return self.get_cost_low(info_dict, action_candidates)

        return self.get_cost_flat(info_dict, action_candidates)
