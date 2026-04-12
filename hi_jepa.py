# Author: Niccolò Caselli 

"""Hierarchical JEPA implementation for top-down temporal planning.

This module defines ``HiJEPA``, a JEPA-style world model with a temporal
hierarchy:

1. Level 3 (strategic): infer a long-range anchor from current state + goal.
2. Level 2 (tactical): infer a midpoint anchor from current state + L3 anchor.
3. Level 1 (reactive): roll out short-horizon latent dynamics with an
   autoregressive predictor conditioned on the midpoint anchor.

The class exposes a ``get_cost(info_dict, action_candidates)`` method compatible
with sample-based planners that expect a cost tensor shaped ``(B, S)``.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    """Detach+clone tensor values, pass through non-tensors unchanged."""
    return v.detach().clone() if torch.is_tensor(v) else v


class HiJEPA(nn.Module):
    """Top-down temporal hierarchy world model.

    Components:
    - ``encoder``: shared visual encoder from images to latent tokens
    - ``pred1``: reactive (Level 1) autoregressive predictor
    - ``pred2``: tactical (Level 2) single-step predictor
    - ``pred3``: strategic (Level 3) single-step predictor
    - ``id2``: tactical inverse dynamics model
    - ``id3``: strategic inverse dynamics model
    - ``action_encoder``: maps raw actions to predictor conditioning space
    - ``projector``/``pred_proj``: latent projection heads (JEPA-compatible)
    """

    def __init__(
        self,
        encoder,
        pred1,
        pred2,
        pred3,
        id2,
        id3,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        """Initialize hierarchical world model components.

        Args:
            encoder: Vision encoder. Called as ``encoder(pixels, interpolate_pos_encoding=True)``
                and expected to return ``last_hidden_state``.
            pred1: Level-1 predictor with signature
                ``pred1(emb: (B,T,D), act_emb: (B,T,A), z_anchor: (B,D)|None) -> (B,T,Dp)``.
            pred2: Level-2 predictor with signature
                ``pred2(z_t: (B,D), a2: (B,A2), z_anchor: (B,D)) -> (B,D)``.
            pred3: Level-3 predictor with signature
                ``pred3(z_t: (B,D), a3: (B,A3)) -> (B,D)``.
            id2: Level-2 inverse dynamics model
                ``id2(z_t: (B,D), z_target: (B,D)) -> (B,A2)``.
            id3: Level-3 inverse dynamics model
                ``id3(z_t: (B,D), z_target: (B,D)) -> (B,A3)``.
            action_encoder: Action embedding module
                ``(B,T,action_dim) -> (B,T,A)``.
            projector: Optional projector applied to encoder CLS outputs.
            pred_proj: Optional projector applied to Level-1 predicted embeddings.
        """
        super().__init__()
        self.encoder = encoder
        self.pred1 = pred1
        self.pred2 = pred2
        self.pred3 = pred3
        self.id2 = id2
        self.id3 = id3
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations/actions into latent embeddings.

        Args:
            info: Dictionary with:
                - ``pixels``: image batch shaped ``(B, T, C, H, W)`` (or compatible tail dims)
                - optional ``action``: action batch shaped ``(B, T, action_dim)``

        Returns:
            dict: Same dictionary enriched with:
                - ``emb``: latent embeddings shaped ``(B, T, D)``
                - optional ``act_emb``: action embeddings shaped ``(B, T, A)``
        """
        pixels = info["pixels"].float()  # (B, T, C, H, W) or (B, T, ...)
        b = pixels.size(0)  # B
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # (B*T, C, H, W) or (B*T, ...)
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # (B*T, D_enc) CLS token
        emb = self.projector(pixels_emb)  # (B*T, D)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)  # (B, T, D)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])  # (B, T, A)

        return info

    def predict(self, emb, act_emb, z_anchor=None):
        """Run Level-1 predictor and projection head.

        Args:
            emb: Context latents ``(B, T, D)``.
            act_emb: Action embeddings aligned with ``emb``, shape ``(B, T, A)``.
            z_anchor: Optional midpoint anchor ``(B, D)`` broadcast into predictor conditioning.

        Returns:
            torch.Tensor: Predicted latents shaped ``(B, T, D)``.
        """
        preds = self.pred1(emb, act_emb, z_anchor=z_anchor)  # (B, T, D_pred)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))  # (B*T, D)
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))  # (B, T, D)
        return preds

    def _compute_anchors(self, z_t, z_g):
        """Compute hierarchical anchors for Levels 3 and 2.

        Args:
            z_t: Current latent state ``(B, D)``.
            z_g: Goal latent state ``(B, D)``.

        Returns:
            dict:
                - ``a3_vec``: strategic macro-action ``(B, A3)``
                - ``z3_anchor``: strategic anchor latent ``(B, D)``
                - ``a2_vec``: tactical macro-action ``(B, A2)``
                - ``z2_anchor``: tactical midpoint latent ``(B, D)``
        """
        a3 = self.id3(z_t, z_g)  # (B, A3)
        z3 = self.pred3(z_t, a3)  # (B, D)
        a2 = self.id2(z_t, z3)  # (B, A2)
        z2 = self.pred2(z_t, a2, z_anchor=z3)  # (B, D)
        return dict(a3_vec=a3, z3_anchor=z3, a2_vec=a2, z2_anchor=z2)

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Roll out Level-1 latents for each candidate action sequence.

        This method reuses the current observation context, tiles it over S
        candidates, and autoregressively predicts future latent states while
        conditioning Level-1 dynamics on a midpoint anchor.

        Args:
            info: Dictionary containing at least:
                - ``pixels``: context pixels ``(B, S, H_ctx, C, H, W)`` or compatible tail dims
                - ``midpoint_anchor``: one of ``(B,D)``, ``(B,1,D)``, ``(B,S,D)``
            action_sequence: Candidate actions ``(B, S, H_plan, action_dim)``.
            history_size: Context window size for autoregressive truncation.

        Returns:
            dict: ``info`` with extra key:
                - ``predicted_emb``: rollout latents ``(B, S, T_rollout, D)``
        """
        assert "pixels" in info, "pixels not in info_dict"
        assert "midpoint_anchor" in info, "midpoint_anchor not in info_dict"

        H = info["pixels"].size(2)  # H_ctx
        B, S, T = action_sequence.shape[:3]  # batch, num_candidates, plan_horizon
        assert T >= H, "action sequence horizon must be >= context horizon"

        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)  # (B,S,H,A), (B,S,T-H,A)
        info["action"] = act_0  # (B,S,H,A) context actions
        n_steps = T - H  # number of autoregressive rollout updates

        # Copy and encode initial info once (using sample index 0), then tile over S.
        _init = {}
        for k, v in info.items():
            if not torch.is_tensor(v):
                continue
            # Fields with sample dimension S are reduced to the first sample.
            if v.ndim >= 2 and v.shape[1] == S:
                _init[k] = v[:, 0]  # (B, ...)
            # Keep already batch-wise fields as-is.
            elif k != "midpoint_anchor":
                _init[k] = v

        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)  # (B,S,H_ctx,D)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # Prepare midpoint anchor per candidate sample.
        midpoint = info["midpoint_anchor"]
        if midpoint.ndim == 2:
            midpoint = midpoint.unsqueeze(1).expand(B, S, -1)  # (B,S,D)
        elif midpoint.ndim == 3 and midpoint.shape[1] == 1:
            midpoint = midpoint.expand(B, S, -1)  # (B,S,D)
        elif midpoint.ndim == 3 and midpoint.shape[1] == S:
            pass  # already (B,S,D)
        else:
            raise ValueError(
                "midpoint_anchor must be shaped as (B,D), (B,1,D), or (B,S,D)"
            )

        # Flatten batch and sample dimensions for rollout.
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()  # (BS,H_ctx,D)
        act = rearrange(act_0, "b s ... -> (b s) ...")  # (BS,H_ctx,A_raw)
        act_future = rearrange(act_future, "b s ... -> (b s) ...")  # (BS,T-H_ctx,A_raw)
        midpoint = rearrange(midpoint, "b s d -> (b s) d")  # (BS,D)

        # Rollout predictor autoregressively for n_steps.
        HS = history_size  # context truncation length
        for t in range(n_steps):
            act_emb = self.action_encoder(act)  # (BS,t_ctx,A)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, D_act)
            pred_emb = self.predict(emb_trunc, act_trunc, z_anchor=midpoint)[:, -1:]  # (BS,1,D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS,t_ctx+1,D)

            next_act = act_future[:, t : t + 1, :]  # (BS,1,A_raw)
            act = torch.cat([act, next_act], dim=1)  # (BS,t_ctx+1,A_raw)

        # Predict one additional step (same behavior as base JEPA rollout).
        act_emb = self.action_encoder(act)  # (BS,T_ctx,A)
        emb_trunc = emb[:, -HS:]  # (BS,HS,D)
        act_trunc = act_emb[:, -HS:]  # (BS,HS,A)
        pred_emb = self.predict(emb_trunc, act_trunc, z_anchor=midpoint)[:, -1:]  # (BS,1,D)
        emb = torch.cat([emb, pred_emb], dim=1)  # (BS,T_rollout,D)

        # Unflatten batch and sample dimensions.
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)  # (B,S,T_rollout,D)
        info["predicted_emb"] = pred_rollout
        return info

    def criterion(self, info_dict: dict):
        """Compute terminal latent-space cost to midpoint anchor.

        Args:
            info_dict: Dictionary with:
                - ``predicted_emb``: rollout latents ``(B,S,T,D)``
                - ``midpoint_anchor``: one of ``(B,D)``, ``(B,1,D)``, ``(B,S,D)``,
                  or pre-expanded ``(B,S,1,D)``

        Returns:
            torch.Tensor: Candidate costs shaped ``(B, S)``.
        """
        pred_emb = info_dict["predicted_emb"]  # (B,S,T,D)
        midpoint = info_dict["midpoint_anchor"]  # (B,D) or (B,S,D)

        if midpoint.ndim == 2:
            target = midpoint.unsqueeze(1).unsqueeze(2)  # (B,1,1,D)
        elif midpoint.ndim == 3:
            if midpoint.shape[1] == pred_emb.shape[1]:
                target = midpoint.unsqueeze(2)  # (B,S,1,D)
            elif midpoint.shape[1] == 1:
                target = midpoint.unsqueeze(2).expand(
                    pred_emb.shape[0], pred_emb.shape[1], 1, pred_emb.shape[-1]
                )
            else:
                raise ValueError("Invalid midpoint_anchor shape for criterion")
        elif midpoint.ndim == 4:
            # Allow pre-expanded target only when it matches (B,S,1,D) semantics.
            if midpoint.shape[0] != pred_emb.shape[0] or midpoint.shape[1] != pred_emb.shape[1]:
                raise ValueError("midpoint_anchor (4D) must match batch/sample dims of predicted_emb")
            if midpoint.shape[2] not in (1, pred_emb.shape[2]):
                raise ValueError("midpoint_anchor (4D) must have time dim 1 or match rollout time")
            if midpoint.shape[3] != pred_emb.shape[3]:
                raise ValueError("midpoint_anchor (4D) latent dim must match predicted_emb")
            target = midpoint
        else:
            raise ValueError("midpoint_anchor has unsupported rank")

        target = target.expand_as(pred_emb[..., -1:, :])

        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            target.detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # sum over (T=1,D) -> (B,S)
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Evaluate candidate actions via 4-phase hierarchical inference.

        Phase 0:
            Encode start and goal observations to latents.
        Phase 1:
            Strategic anchor from ``id3`` + ``pred3``.
        Phase 2:
            Tactical midpoint from ``id2`` + ``pred2``.
        Phase 3:
            Level-1 short-horizon rollout and midpoint-based terminal cost.

        Args:
            info_dict: Planning context dictionary. Must include:
                - ``pixels``: rollout context with shape compatible with
                  ``(B,S,H_ctx,...)``.
                - ``goal``: goal observations with sample axis, shape compatible
                  with ``(B,S,T_goal,...)`` where ``[:,0]`` is valid.
            action_candidates: Candidate actions ``(B,S,H_plan,action_dim)``.

        Returns:
            torch.Tensor: Cost matrix shaped ``(B,S)``.
        """
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        # Phase 0a: build a goal dict in a JEPA-compatible way.
        # We intentionally mirror the base JEPA behavior:
        # - start from tensor keys reduced on sample axis ([:, 0])
        # - route goal pixels through the "pixels" key
        # - remap goal_* auxiliary keys (e.g., goal_state -> state) when present
        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]  # planner goal observation becomes encoder input

        for k in list(goal.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        # Action is not needed to encode the goal state and may have incompatible shape.
        goal.pop("action", None)

        # Phase 0b: encode goal latent.
        goal = self.encode(goal)  # goal["emb"]: (B, T_goal, D)
        z_g = goal["emb"][:, -1]  # (B, D) use final goal timestep when T_goal > 1

        # Phase 0c: encode current state from last context frame.
        # info_dict["pixels"] is expected to be (B, S, H_ctx, ...), so:
        # - select candidate axis 0 (all candidates share same current context)
        # - select last context frame via -1:
        start_pixels = info_dict["pixels"][:, 0, -1:]  # (B, 1, ...)
        start = self.encode({"pixels": start_pixels})
        z_t = start["emb"][:, 0]  # (B, D)

        # Phase 1 + 2: strategic and tactical anchors
        anchors = self._compute_anchors(z_t, z_g)
        info_dict["strategic_anchor"] = anchors["z3_anchor"]
        info_dict["midpoint_anchor"] = anchors["z2_anchor"]
        info_dict["goal_emb"] = anchors["z2_anchor"]  # compatibility

        # Phase 3: short-horizon rollout with midpoint anchor
        info_dict = self.rollout(info_dict, action_candidates)
        cost = self.criterion(info_dict)
        return cost
