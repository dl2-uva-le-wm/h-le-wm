from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from copy import deepcopy

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hi_jepa import HiJEPA
from hi_policy import HierarchicalWorldModelPolicy, calibrate_latent_prior


class _EncOut:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class DummyVisionEncoder(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, pixels: torch.Tensor, interpolate_pos_encoding: bool = True):
        # pixels: (B*T, C, H, W) -> cls token embedding (B*T, D)
        flat = pixels.float().flatten(start_dim=1)
        cls = flat.mean(dim=1, keepdim=True).repeat(1, self.emb_dim)
        cls = cls + self.bias
        return _EncOut(last_hidden_state=cls[:, None, :])


class AdditivePredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, 32, dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return x + c


class IdentityActionEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.float())


class MeanLatentActionEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.output_proj = nn.Linear(input_dim, latent_dim, bias=False)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.weight[: min(input_dim, latent_dim), : min(input_dim, latent_dim)] = torch.eye(
                min(input_dim, latent_dim)
            )

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        if action_mask is not None:
            mask = action_mask.to(x.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (x * mask).sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)
        return self.output_proj(pooled)


class ScaleProjection(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def make_test_hijepa(dim: int = 4) -> HiJEPA:
    return HiJEPA(
        encoder=DummyVisionEncoder(dim),
        low_predictor=AdditivePredictor(dim),
        action_encoder=IdentityActionEncoder(dim),
        high_predictor=AdditivePredictor(dim),
        latent_action_encoder=MeanLatentActionEncoder(dim, dim),
        macro_to_condition=nn.Identity(),
        projector=nn.Identity(),
        low_pred_proj=nn.Identity(),
        high_pred_proj=nn.Identity(),
    )


def test_split_projection_heads_are_used_independently():
    dim = 4
    model = HiJEPA(
        encoder=DummyVisionEncoder(dim),
        low_predictor=AdditivePredictor(dim),
        action_encoder=IdentityActionEncoder(dim),
        high_predictor=AdditivePredictor(dim),
        latent_action_encoder=MeanLatentActionEncoder(dim, dim),
        macro_to_condition=nn.Identity(),
        projector=nn.Identity(),
        low_pred_proj=ScaleProjection(2.0),
        high_pred_proj=ScaleProjection(5.0),
    )

    emb = torch.ones(2, 3, dim)
    cond = torch.zeros(2, 3, dim)
    low_out = model.predict_low(emb, cond)
    high_out = model.predict_high(emb, cond)

    assert torch.allclose(low_out, emb * 2.0)
    assert torch.allclose(high_out, emb * 5.0)


def test_copy_init_projection_weights_are_equal_but_not_shared():
    dim = 4
    low_proj = nn.Linear(dim, dim, bias=True)
    with torch.no_grad():
        low_proj.weight.uniform_(-0.1, 0.1)
        low_proj.bias.uniform_(-0.1, 0.1)
    high_proj = deepcopy(low_proj)

    model = HiJEPA(
        encoder=DummyVisionEncoder(dim),
        low_predictor=AdditivePredictor(dim),
        action_encoder=IdentityActionEncoder(dim),
        high_predictor=AdditivePredictor(dim),
        latent_action_encoder=MeanLatentActionEncoder(dim, dim),
        macro_to_condition=nn.Identity(),
        projector=nn.Identity(),
        low_pred_proj=low_proj,
        high_pred_proj=high_proj,
    )

    assert torch.allclose(model.low_pred_proj.weight, model.high_pred_proj.weight)
    assert torch.allclose(model.low_pred_proj.bias, model.high_pred_proj.bias)
    assert model.low_pred_proj.weight.data_ptr() != model.high_pred_proj.weight.data_ptr()
    assert model.low_pred_proj.bias.data_ptr() != model.high_pred_proj.bias.data_ptr()


def test_freezing_low_projection_keeps_high_projection_trainable():
    dim = 4
    model = HiJEPA(
        encoder=DummyVisionEncoder(dim),
        low_predictor=AdditivePredictor(dim),
        action_encoder=IdentityActionEncoder(dim),
        high_predictor=AdditivePredictor(dim),
        latent_action_encoder=MeanLatentActionEncoder(dim, dim),
        macro_to_condition=nn.Identity(),
        projector=nn.Identity(),
        low_pred_proj=nn.Linear(dim, dim),
        high_pred_proj=nn.Linear(dim, dim),
    )

    model.freeze_low_level(
        freeze_encoder=False,
        freeze_low_predictor=False,
        freeze_action_encoder=False,
        freeze_projector=False,
        freeze_low_pred_proj=True,
        freeze_high_pred_proj=False,
    )

    assert not any(p.requires_grad for p in model.low_pred_proj.parameters())
    assert all(p.requires_grad for p in model.high_pred_proj.parameters())


def test_hijepa_rollout_shapes():
    model = make_test_hijepa(dim=4)

    z_init = torch.zeros(2, 4)
    latent_actions = torch.randn(2, 3, 5, 4)
    high = model.rollout_high(z_init, latent_actions)
    assert high.shape == (2, 3, 5, 4)

    z_hist = torch.zeros(2, 3, 1, 4)
    low_actions = torch.randn(2, 3, 6, 4)
    low = model.rollout_low(z_hist, None, low_actions)
    assert low.shape == (2, 3, 6, 4)


def test_get_cost_high_zero_for_matching_goal():
    torch.manual_seed(0)
    model = make_test_hijepa(dim=4)
    z_init = torch.randn(2, 4)
    actions = torch.randn(2, 3, 4, 4)
    pred = model.rollout_high(z_init, actions)
    z_goal = pred[:, :, -1, :].detach()

    info = {"planner_level": "high", "z_init": z_init, "z_goal": z_goal}
    cost = model.get_cost(info, actions)
    assert cost.shape == (2, 3)
    assert torch.allclose(cost, torch.zeros_like(cost), atol=1e-6)


def test_get_cost_low_prefers_matching_subgoal():
    torch.manual_seed(1)
    model = make_test_hijepa(dim=4)
    z_hist = torch.zeros(2, 3, 1, 4)
    actions = torch.randn(2, 3, 5, 4)
    pred = model.rollout_low(z_hist, None, actions)
    z_sub = pred[:, :, -1, :].detach()

    info_match = {
        "planner_level": "low",
        "z_hist": z_hist,
        "a_hist": torch.zeros(2, 3, 1, 4),
        "z_subgoal": z_sub,
    }
    info_shift = {
        **info_match,
        "z_subgoal": z_sub + 1.0,
    }
    cost_match = model.get_cost(info_match, actions)
    cost_shift = model.get_cost(info_shift, actions)
    assert cost_match.shape == (2, 3)
    assert torch.isfinite(cost_match).all()
    assert torch.isfinite(cost_shift).all()
    assert (cost_shift > cost_match).all()


class FakeDataset:
    def __init__(
        self,
        actions: np.ndarray,
        *,
        episode_idx: np.ndarray | None = None,
        step_idx: np.ndarray | None = None,
    ):
        self._actions = actions
        self._episode_idx = episode_idx
        self._step_idx = step_idx
        cols = ["action"]
        if episode_idx is not None:
            cols.append("episode_idx")
        if step_idx is not None:
            cols.append("step_idx")
        self.column_names = cols

    def get_col_data(self, name: str):
        if name == "action":
            return self._actions
        if name == "episode_idx":
            if self._episode_idx is None:
                raise KeyError(name)
            return self._episode_idx
        if name == "step_idx":
            if self._step_idx is None:
                raise KeyError(name)
            return self._step_idx
        raise KeyError(name)


def test_calibrate_latent_prior_bounds_deterministic_and_ordered():
    torch.manual_seed(0)
    model = make_test_hijepa(dim=4)
    actions = np.random.default_rng(7).normal(size=(500, 4)).astype(np.float32)
    dataset = FakeDataset(actions)
    cfg = {
        "enabled": True,
        "num_chunks": 200,
        "min_chunks_for_stats": 32,
        "chunk_len": 5,
        "lower_q": 10.0,
        "upper_q": 90.0,
        "margin_ratio": 0.1,
        "clamp_abs": 2.0,
        "fallback_abs": 1.0,
    }

    b1 = calibrate_latent_prior(model=model, dataset=dataset, cfg=cfg, seed=123)
    b2 = calibrate_latent_prior(model=model, dataset=dataset, cfg=cfg, seed=123)

    assert b1["low"].shape == (4,)
    assert b1["high"].shape == (4,)
    assert np.isfinite(b1["low"]).all()
    assert np.isfinite(b1["high"]).all()
    assert np.all(b1["high"] > b1["low"])
    np.testing.assert_allclose(b1["low"], b2["low"])
    np.testing.assert_allclose(b1["high"], b2["high"])


def test_calibrate_latent_prior_fallback_for_short_dataset():
    model = make_test_hijepa(dim=4)
    actions = np.random.default_rng(3).normal(size=(3, 4)).astype(np.float32)
    dataset = FakeDataset(actions)
    cfg = {
        "enabled": True,
        "num_chunks": 100,
        "min_chunks_for_stats": 32,
        "chunk_len": 8,
        "fallback_abs": 1.5,
    }
    b = calibrate_latent_prior(model=model, dataset=dataset, cfg=cfg, seed=0)
    np.testing.assert_allclose(b["low"], -1.5)
    np.testing.assert_allclose(b["high"], 1.5)
    assert int(b["num_chunks"]) == 0


def test_calibrate_latent_prior_respects_episode_boundaries():
    model = make_test_hijepa(dim=4)
    # Two episodes of length 3 each. Global flattening would allow chunk_len=4,
    # but episode-aware sampling must reject all such starts.
    actions = np.random.default_rng(9).normal(size=(6, 4)).astype(np.float32)
    episode_idx = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    step_idx = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    dataset = FakeDataset(actions, episode_idx=episode_idx, step_idx=step_idx)
    cfg = {
        "enabled": True,
        "num_chunks": 128,
        "min_chunks_for_stats": 16,
        "chunk_len": 4,
        "fallback_abs": 1.25,
    }

    b = calibrate_latent_prior(model=model, dataset=dataset, cfg=cfg, seed=7)
    np.testing.assert_allclose(b["low"], -1.25)
    np.testing.assert_allclose(b["high"], 1.25)
    assert int(b["num_chunks"]) == 0


@dataclass
class DummyPlanConfig:
    horizon: int
    receding_horizon: int
    action_block: int
    warm_start: bool = True


class DummyActionSpace:
    def __init__(self, shape):
        self.shape = shape


class DummyEnv:
    def __init__(self, num_envs: int, action_dim: int):
        self.num_envs = num_envs
        self.action_space = DummyActionSpace((num_envs, action_dim))


class FakeSolver:
    def __init__(self):
        self.calls = []
        self.configure_calls = []
        self._n_envs = 0
        self._config = None
        self._action_dim = 0

    def configure(self, *, action_space, n_envs: int, config):
        self.configure_calls.append((action_space, n_envs, config))
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:])) * int(config.action_block)

    @property
    def action_dim(self):
        return self._action_dim

    def __call__(self, info_dict: dict, init_action: torch.Tensor | None = None):
        self.calls.append({"info": info_dict, "init_action": init_action})
        horizon = int(self._config.horizon)
        actions = torch.zeros(self._n_envs, horizon, self._action_dim, dtype=torch.float32)
        actions += float(len(self.calls))
        return {"actions": actions}


class FakePolicyModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.latent_action_encoder = MeanLatentActionEncoder(dim, dim)
        self.dim = dim

    def encode(self, info: dict, *, encode_actions: bool = False) -> dict:
        px = info["pixels"].float()  # (B, T, C, H, W)
        emb = px.flatten(start_dim=2).mean(dim=-1, keepdim=True).repeat(1, 1, self.dim)
        info["emb"] = emb
        return info

    def rollout_high(self, z_init: torch.Tensor, latent_actions: torch.Tensor) -> torch.Tensor:
        if latent_actions.ndim == 3:
            latent_actions = latent_actions.unsqueeze(1)
        return latent_actions.cumsum(dim=2) + z_init.unsqueeze(1).unsqueeze(2)


def test_hierarchical_policy_replans_high_and_warm_starts_low():
    model = FakePolicyModel(dim=3)
    high_solver = FakeSolver()
    low_solver = FakeSolver()
    high_cfg = DummyPlanConfig(horizon=2, receding_horizon=1, action_block=1, warm_start=True)
    low_cfg = DummyPlanConfig(horizon=2, receding_horizon=1, action_block=2, warm_start=True)

    policy = HierarchicalWorldModelPolicy(
        model=model,
        high_solver=high_solver,
        low_solver=low_solver,
        high_config=high_cfg,
        low_config=low_cfg,
        macro_replan_interval=2,
        process={},
        transform={},
    )
    env = DummyEnv(num_envs=2, action_dim=3)
    policy.set_env(env)

    info = {
        "pixels": np.random.default_rng(0).normal(size=(2, 1, 8, 8, 3)).astype(np.float32),
        "goal": np.random.default_rng(1).normal(size=(2, 1, 8, 8, 3)).astype(np.float32),
    }

    a1 = policy.get_action(info)
    a2 = policy.get_action(info)
    a3 = policy.get_action(info)

    assert a1.shape == (2, 3)
    assert a2.shape == (2, 3)
    assert a3.shape == (2, 3)

    # low-level replans only when buffer is empty: call 1 and 3
    assert len(low_solver.calls) == 2
    # high-level replans first call, then every k=2 steps (call 3)
    assert len(high_solver.calls) == 2
    # warm-start applied on second low solve
    assert low_solver.calls[0]["init_action"] is None
    assert low_solver.calls[1]["init_action"] is not None
