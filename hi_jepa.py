import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    """Detach+clone tensor values, pass through non-tensors unchanged."""
    return v.detach().clone() if torch.is_tensor(v) else v


class HiJEPA(nn.Module):
    """Hierarchical JEPA world model with runtime-selectable 2 or 3 levels."""

    def __init__(
        self,
        encoder,
        pred1,
        pred2,
        id2,
        action_encoder,
        pred3=None,
        id3=None,
        num_levels: int = 3,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()
        self.num_levels = int(num_levels)
        if self.num_levels not in (2, 3):
            raise ValueError(f"num_levels must be 2 or 3, got {self.num_levels}")
        if self.num_levels == 3 and (pred3 is None or id3 is None):
            raise ValueError("num_levels=3 requires both pred3 and id3")

        self.encoder = encoder
        self.pred1 = pred1
        self.pred2 = pred2
        self.pred3 = pred3
        self.id2 = id2
        self.id3 = id3
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def _require_level3(self):
        if self.pred3 is None or self.id3 is None:
            raise RuntimeError("Level-3 modules (pred3/id3) are required but not available")

    def encode(self, info):
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb, z_anchor=None):
        preds = self.pred1(emb, act_emb, z_anchor=z_anchor)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def _compute_midpoint_anchor(self, z_t, z_g):
        """Compute midpoint anchor for rollout from current and goal latents."""
        if self.num_levels == 3:
            self._require_level3()
            a3 = self.id3(z_t, z_g)
            z3 = self.pred3(z_t, a3)
            a2 = self.id2(z_t, z3)
            z2 = self.pred2(z_t, a2, z_anchor=z3)
            return {"a3_vec": a3, "z3_anchor": z3, "a2_vec": a2, "z2_anchor": z2}

        a2 = self.id2(z_t, z_g)
        z2 = self.pred2(z_t, a2, z_anchor=z_g)
        return {"a2_vec": a2, "z2_anchor": z2}

    def rollout(self, info, action_sequence, history_size: int = 3):
        assert "pixels" in info, "pixels not in info_dict"
        assert "midpoint_anchor" in info, "midpoint_anchor not in info_dict"

        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        assert T >= H, "action sequence horizon must be >= context horizon"

        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {}
        for k, v in info.items():
            if not torch.is_tensor(v):
                continue
            if v.ndim >= 2 and v.shape[1] == S:
                _init[k] = v[:, 0]
            elif k != "midpoint_anchor":
                _init[k] = v

        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        midpoint = info["midpoint_anchor"]
        if midpoint.ndim == 2:
            midpoint = midpoint.unsqueeze(1).expand(B, S, -1)
        elif midpoint.ndim == 3 and midpoint.shape[1] == 1:
            midpoint = midpoint.expand(B, S, -1)
        elif midpoint.ndim == 3 and midpoint.shape[1] == S:
            pass
        else:
            raise ValueError("midpoint_anchor must be shaped as (B,D), (B,1,D), or (B,S,D)")

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")
        midpoint = rearrange(midpoint, "b s d -> (b s) d")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc, z_anchor=midpoint)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc, z_anchor=midpoint)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info

    def criterion(self, info_dict: dict):
        pred_emb = info_dict["predicted_emb"]
        midpoint = info_dict["midpoint_anchor"]

        if midpoint.ndim == 2:
            target = midpoint.unsqueeze(1).unsqueeze(2)
        elif midpoint.ndim == 3:
            if midpoint.shape[1] == pred_emb.shape[1]:
                target = midpoint.unsqueeze(2)
            elif midpoint.shape[1] == 1:
                target = midpoint.unsqueeze(2).expand(
                    pred_emb.shape[0], pred_emb.shape[1], 1, pred_emb.shape[-1]
                )
            else:
                raise ValueError("Invalid midpoint_anchor shape for criterion")
        elif midpoint.ndim == 4:
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
        ).sum(dim=tuple(range(2, pred_emb.ndim)))
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in list(goal.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action", None)
        goal = self.encode(goal)
        z_g = goal["emb"][:, -1]

        start_pixels = info_dict["pixels"][:, 0, -1:]
        start = self.encode({"pixels": start_pixels})
        z_t = start["emb"][:, 0]

        anchors = self._compute_midpoint_anchor(z_t, z_g)
        info_dict["midpoint_anchor"] = anchors["z2_anchor"]
        info_dict["goal_emb"] = anchors["z2_anchor"]

        if "z3_anchor" in anchors:
            info_dict["strategic_anchor"] = anchors["z3_anchor"]
        else:
            info_dict.pop("strategic_anchor", None)

        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)
