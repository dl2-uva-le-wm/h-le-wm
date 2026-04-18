from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hi_jepa import HiJEPA


def _load_hi_train_functions():
    """Load selected functions from hi_train.py without importing heavy deps."""
    src_path = Path(__file__).resolve().parents[1] / "hi_train.py"
    source = src_path.read_text()
    mod = ast.parse(source)

    wanted = {
        "gather_waypoint_embeddings",
        "build_action_chunks",
        "build_action_chunks_batched",
        "is_p2_frozen_optimization_enabled",
        "build_p2_frozen_waypoint_collate",
        "hi_lejepa_forward",
        "hi_lejepa_forward_p2_frozen",
        "clone_projection_head",
    }
    chunks = []
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            chunks.append(ast.get_source_segment(source, node))

    ns = {"torch": torch, "deepcopy": deepcopy, "default_collate": default_collate}
    exec("\n\n".join(chunks), ns)
    return ns


_HI_NS = _load_hi_train_functions()
GATHER_WAYPOINT_EMBEDDINGS = _HI_NS["gather_waypoint_embeddings"]
BUILD_ACTION_CHUNKS = _HI_NS["build_action_chunks"]
BUILD_ACTION_CHUNKS_BATCHED = _HI_NS["build_action_chunks_batched"]
HI_LEJEPA_FORWARD = _HI_NS["hi_lejepa_forward"]
HI_LEJEPA_FORWARD_P2_FROZEN = _HI_NS["hi_lejepa_forward_p2_frozen"]
IS_P2_FROZEN_OPT_ENABLED = _HI_NS["is_p2_frozen_optimization_enabled"]
BUILD_P2_FROZEN_WAYPOINT_COLLATE = _HI_NS["build_p2_frozen_waypoint_collate"]
CLONE_PROJECTION_HEAD = _HI_NS["clone_projection_head"]


class Node(dict):
    """Small config-like dict with dot-attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class TinyEncoder(nn.Module):
    def __init__(self, in_channels: int, img_hw: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(in_channels * img_hw * img_hw, dim)

    def forward(self, x: torch.Tensor, interpolate_pos_encoding: bool = True):
        flat = x.reshape(x.size(0), -1)
        cls = self.proj(flat)
        return SimpleNamespace(last_hidden_state=cls.unsqueeze(1))


class DummyTwoInputPredictor(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + 0.0 * y


class DummyModel:
    def __init__(self, *, embed_dim: int = 8, latent_dim: int = 5):
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.encode_calls = 0
        self.encode_selected_calls = 0
        self.last_z_context_shape = None
        self.last_macro_shape = None
        self.last_z_pred_shape = None
        self.last_flat_actions_shape = None
        self.last_flat_mask_shape = None

    def encode(self, batch: dict, *, encode_actions: bool = True):
        self.encode_calls += 1
        pixels = batch["pixels"]
        b, t = pixels.shape[:2]
        emb = (
            torch.arange(t, device=pixels.device, dtype=pixels.dtype)
            .view(1, t, 1)
            .expand(b, t, self.embed_dim)
            .clone()
        )
        out = {"emb": emb}
        if encode_actions:
            out["act_emb"] = torch.zeros_like(emb)
        return out

    def encode_selected_frames(self, pixels: torch.Tensor, frame_indices: torch.Tensor):
        self.encode_selected_calls += 1
        b, n = frame_indices.shape
        return (
            frame_indices.to(device=pixels.device, dtype=pixels.dtype)
            .view(b, n, 1)
            .expand(b, n, self.embed_dim)
            .clone()
        )

    def encode_macro_actions(self, action_chunks: torch.Tensor, action_mask: torch.Tensor):
        self.last_flat_actions_shape = tuple(action_chunks.shape)
        self.last_flat_mask_shape = tuple(action_mask.shape)
        valid = action_mask.unsqueeze(-1).to(dtype=action_chunks.dtype)
        summed = (action_chunks * valid).sum(dim=1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        out = torch.zeros(
            action_chunks.size(0),
            self.latent_dim,
            device=action_chunks.device,
            dtype=action_chunks.dtype,
        )
        take = min(self.latent_dim, pooled.size(-1))
        out[:, :take] = pooled[:, :take]
        return out

    def predict_high(self, emb: torch.Tensor, macro_actions: torch.Tensor):
        self.last_z_context_shape = tuple(emb.shape)
        self.last_macro_shape = tuple(macro_actions.shape)

        macro_proj = torch.zeros(
            emb.size(0),
            emb.size(1),
            emb.size(2),
            device=emb.device,
            dtype=emb.dtype,
        )
        take = min(emb.size(-1), macro_actions.size(-1))
        macro_proj[:, :, :take] = macro_actions[:, :, :take]
        pred = emb + macro_proj
        self.last_z_pred_shape = tuple(pred.shape)
        return pred

    def predict_low(self, emb: torch.Tensor, act_emb: torch.Tensor):
        return emb + 0.0 * act_emb


class DummyModule:
    def __init__(self, model: DummyModel):
        self.model = model
        self.logged = []

    def sigreg(self, emb_tbd: torch.Tensor):
        return emb_tbd.pow(2).mean()

    def log_dict(self, metrics: dict, on_step: bool, sync_dist: bool):
        self.logged.append((metrics, on_step, sync_dist))


def _sample_waypoints_stub(_cfg, batch_size: int, seq_len: int, device):
    assert seq_len >= 11
    base = torch.tensor([2, 4, 6, 8, 10], device=device, dtype=torch.long)
    waypoints = base.unsqueeze(0).expand(batch_size, -1).clone()
    gaps = waypoints[:, 1:] - waypoints[:, :-1]
    return waypoints, gaps


def _make_cfg(*, train_low_level: bool, sigreg_weight: float):
    return Node(
        training=Node(train_low_level=train_low_level),
        pretrained_low_level=Node(
            enabled=True,
            freeze=Node(
                encoder=True,
                low_level_predictor=True,
                low_level_action_encoder=True,
                projector=True,
                low_pred_proj=True,
            ),
        ),
        loss=Node(
            alpha=0.0,
            beta=1.0,
            sigreg=Node(weight=float(sigreg_weight)),
        ),
        wm=Node(
            history_size=3,
            num_preds=1,
            high_level=Node(waypoints=Node(num=5)),
        ),
    )


def test_encode_selected_frames_matches_full_encode_gather():
    torch.manual_seed(0)
    b, t, c, h, w = 2, 12, 3, 4, 4
    d = 6
    pixels = torch.randn(b, t, c, h, w)
    indices = torch.tensor([[2, 4, 8], [3, 7, 10]], dtype=torch.long)

    model = HiJEPA(
        encoder=TinyEncoder(in_channels=c, img_hw=h, dim=d),
        low_predictor=DummyTwoInputPredictor(),
        action_encoder=nn.Identity(),
        high_predictor=DummyTwoInputPredictor(),
        latent_action_encoder=nn.Identity(),
        macro_to_condition=nn.Identity(),
        projector=nn.Identity(),
        low_pred_proj=nn.Identity(),
        high_pred_proj=nn.Identity(),
    )

    full = model.encode({"pixels": pixels}, encode_actions=False)["emb"]
    gathered = GATHER_WAYPOINT_EMBEDDINGS(full, indices)
    selected = model.encode_selected_frames(pixels, indices)

    assert selected.shape == (b, indices.size(1), d)
    assert torch.allclose(selected, gathered, atol=1e-6)


def test_build_action_chunks_batched_matches_reference_loop():
    torch.manual_seed(1)
    b, t, a, k = 4, 20, 6, 4
    actions = torch.randn(b, t, a)

    starts = torch.tensor(
        [
            [1, 4, 5, 10],
            [2, 7, 8, 11],
            [0, 6, 9, 12],
            [3, 5, 7, 13],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor(
        [
            [2, 5, 3, 4],
            [3, 2, 6, 2],
            [4, 3, 2, 5],
            [2, 4, 3, 3],
        ],
        dtype=torch.long,
    )
    ends = starts + lengths

    batched_chunks, batched_mask = BUILD_ACTION_CHUNKS_BATCHED(actions, starts, ends)
    assert batched_chunks.shape[:2] == (b, k)
    assert batched_mask.shape[:2] == (b, k)

    for step in range(k):
        ref_chunks, ref_mask = BUILD_ACTION_CHUNKS(actions, starts[:, step], ends[:, step])
        ref_len = ref_chunks.size(1)

        assert torch.equal(batched_mask[:, step, :ref_len], ref_mask)
        assert torch.allclose(batched_chunks[:, step, :ref_len], ref_chunks)

        if batched_chunks.size(2) > ref_len:
            assert not batched_mask[:, step, ref_len:].any()
            assert torch.allclose(
                batched_chunks[:, step, ref_len:],
                torch.zeros_like(batched_chunks[:, step, ref_len:]),
            )


def test_build_action_chunks_batched_rejects_non_positive_lengths():
    actions = torch.randn(2, 10, 3)
    starts = torch.tensor([[1, 3], [2, 4]], dtype=torch.long)
    ends = starts.clone()

    with pytest.raises(ValueError, match="positive length"):
        BUILD_ACTION_CHUNKS_BATCHED(actions, starts, ends)


def test_hi_lejepa_forward_fast_path_shapes_and_single_macro_encode():
    HI_LEJEPA_FORWARD.__globals__["sample_waypoints"] = _sample_waypoints_stub

    model = DummyModel(embed_dim=8, latent_dim=5)
    module = DummyModule(model)
    cfg = _make_cfg(train_low_level=False, sigreg_weight=0.0)

    batch = {
        "pixels": torch.randn(2, 12, 3, 4, 4),
        "action": torch.randn(2, 12, 6),
    }

    out = HI_LEJEPA_FORWARD(module, batch, "train", cfg)

    assert model.encode_selected_calls == 1
    assert model.encode_calls == 0
    assert model.last_z_context_shape == (2, 4, 8)
    assert model.last_macro_shape == (2, 4, 5)
    assert model.last_z_pred_shape == (2, 4, 8)
    assert model.last_flat_actions_shape[0] == 2 * 4
    assert model.last_flat_mask_shape[0] == 2 * 4
    assert torch.isfinite(out["loss"])


def test_hi_lejepa_forward_sigreg_guard_uses_full_sequence_encode():
    HI_LEJEPA_FORWARD.__globals__["sample_waypoints"] = _sample_waypoints_stub

    model = DummyModel(embed_dim=8, latent_dim=5)
    module = DummyModule(model)
    cfg = _make_cfg(train_low_level=False, sigreg_weight=0.3)

    batch = {
        "pixels": torch.randn(2, 12, 3, 4, 4),
        "action": torch.randn(2, 12, 6),
    }

    out = HI_LEJEPA_FORWARD(module, batch, "train", cfg)

    assert model.encode_calls == 1
    assert model.encode_selected_calls == 0
    assert out["sigreg_loss"].item() > 0.0


def test_hi_lejepa_forward_smoke_multiple_steps_logs_metrics():
    HI_LEJEPA_FORWARD.__globals__["sample_waypoints"] = _sample_waypoints_stub

    model = DummyModel(embed_dim=8, latent_dim=5)
    module = DummyModule(model)
    cfg = _make_cfg(train_low_level=False, sigreg_weight=0.0)

    for _ in range(3):
        batch = {
            "pixels": torch.randn(2, 12, 3, 4, 4),
            "action": torch.randn(2, 12, 6),
        }
        out = HI_LEJEPA_FORWARD(module, batch, "train", cfg)
        assert torch.isfinite(out["loss"])

    assert len(module.logged) == 3


def test_is_p2_frozen_optimization_enabled_matches_expected_modes():
    cfg = _make_cfg(train_low_level=False, sigreg_weight=0.0)
    assert IS_P2_FROZEN_OPT_ENABLED(cfg)

    cfg_low = _make_cfg(train_low_level=True, sigreg_weight=0.0)
    assert not IS_P2_FROZEN_OPT_ENABLED(cfg_low)

    cfg_sig = _make_cfg(train_low_level=False, sigreg_weight=0.2)
    assert not IS_P2_FROZEN_OPT_ENABLED(cfg_sig)

    cfg_no_pretrain = _make_cfg(train_low_level=False, sigreg_weight=0.0)
    cfg_no_pretrain.pretrained_low_level.enabled = False
    assert not IS_P2_FROZEN_OPT_ENABLED(cfg_no_pretrain)

    cfg_not_frozen = _make_cfg(train_low_level=False, sigreg_weight=0.0)
    cfg_not_frozen.pretrained_low_level.freeze.low_pred_proj = False
    assert not IS_P2_FROZEN_OPT_ENABLED(cfg_not_frozen)


def test_build_p2_frozen_waypoint_collate_slices_pixels_and_attaches_waypoints():
    BUILD_P2_FROZEN_WAYPOINT_COLLATE.__globals__["sample_waypoints"] = _sample_waypoints_stub

    cfg = _make_cfg(train_low_level=False, sigreg_weight=0.0)
    calls = {"n": 0}

    def pixel_preprocessor(sample: dict):
        calls["n"] += 1
        return {"pixels": sample["pixels"].float()}

    collate = BUILD_P2_FROZEN_WAYPOINT_COLLATE(cfg, pixel_preprocessor)
    samples = [
        {
            "pixels": torch.randint(0, 255, (12, 3, 4, 4), dtype=torch.uint8),
            "action": torch.randn(12, 6),
            "proprio": torch.randn(12, 2),
        },
        {
            "pixels": torch.randint(0, 255, (12, 3, 4, 4), dtype=torch.uint8),
            "action": torch.randn(12, 6),
            "proprio": torch.randn(12, 2),
        },
    ]
    batch = collate(samples)

    assert calls["n"] == 2
    assert batch["pixels"].shape == (2, 5, 3, 4, 4)
    assert batch["waypoints"].shape == (2, 5)
    assert batch["action"].shape == (2, 12, 6)


def test_hi_lejepa_forward_p2_frozen_uses_presliced_pixels_and_waypoints():
    model = DummyModel(embed_dim=8, latent_dim=5)
    module = DummyModule(model)
    cfg = _make_cfg(train_low_level=False, sigreg_weight=0.0)

    batch = {
        "pixels": torch.randn(2, 5, 3, 4, 4),
        "action": torch.randn(2, 12, 6),
        "waypoints": torch.tensor([[2, 4, 6, 8, 10], [2, 4, 6, 8, 10]], dtype=torch.long),
    }

    out = HI_LEJEPA_FORWARD_P2_FROZEN(module, batch, "train", cfg)

    assert model.encode_calls == 1
    assert model.encode_selected_calls == 0
    assert model.last_z_context_shape == (2, 4, 8)
    assert model.last_macro_shape == (2, 4, 5)
    assert model.last_z_pred_shape == (2, 4, 8)
    assert torch.isfinite(out["loss"])
    assert out["l1_pred_loss"].item() == 0.0
    assert out["sigreg_loss"].item() == 0.0


def test_clone_projection_head_copies_weights_without_sharing_storage():
    low = nn.Linear(8, 8, bias=True)
    with torch.no_grad():
        low.weight.uniform_(-0.5, 0.5)
        low.bias.uniform_(-0.5, 0.5)

    high = CLONE_PROJECTION_HEAD(low)

    assert torch.allclose(low.weight, high.weight)
    assert torch.allclose(low.bias, high.bias)
    assert low.weight.data_ptr() != high.weight.data_ptr()
    assert low.bias.data_ptr() != high.bias.data_ptr()
