from __future__ import annotations

import ast
import warnings
from pathlib import Path
import pytest
import torch


def _load_sampling_functions():
    """Load sampling functions from hi_waypoint_sampling.py without heavy deps."""
    src_path = Path(__file__).resolve().parents[1] / "hi_waypoint_sampling.py"
    source = src_path.read_text()
    mod = ast.parse(source)

    wanted = {
        "_sample_random_middle",
        "_sample_random_sorted",
        "_sample_fixed_stride",
        "sample_waypoints",
    }
    chunks = []
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            chunks.append(ast.get_source_segment(source, node))

    ns = {"torch": torch, "warnings": warnings}
    exec("\n\n".join(chunks), ns)
    return ns["sample_waypoints"]


SAMPLE_WAYPOINTS = _load_sampling_functions()


class Node(dict):
    """Small config-like dict with dot-attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _make_cfg(
    *,
    strategy: str,
    num_waypoints: int,
    min_stride: int,
    max_span: int,
    stride: int = -1,
    history_size: int = 3,
    beta_alpha: float = 2.0,
    beta_beta: float = 2.0,
):
    return Node(
        wm=Node(
            history_size=history_size,
            high_level=Node(
                waypoints=Node(
                    num=num_waypoints,
                    strategy=strategy,
                    min_stride=min_stride,
                    max_span=max_span,
                    stride=stride,
                    beta_alpha=beta_alpha,
                    beta_beta=beta_beta,
                )
            ),
        )
    )


def _assert_monotonic_and_valid(
    waypoints: torch.Tensor,
    *,
    seq_len: int,
    min_stride: int | None = None,
):
    assert waypoints.ndim == 2
    assert (waypoints[:, 1:] > waypoints[:, :-1]).all()
    assert (waypoints[:, -1] < seq_len).all()
    if min_stride is not None:
        gaps = waypoints[:, 1:] - waypoints[:, :-1]
        assert (gaps >= min_stride).all()


def test_random_middle_sampling_constraints():
    torch.manual_seed(0)
    cfg = _make_cfg(
        strategy="random_middle",
        num_waypoints=3,
        min_stride=2,
        max_span=15,
        history_size=3,
    )
    batch_size = 8
    seq_len = 40

    waypoints, gaps = SAMPLE_WAYPOINTS(cfg, batch_size=batch_size, seq_len=seq_len, device="cpu")

    assert waypoints.shape == (batch_size, 3)
    assert gaps.shape == (batch_size, 2)
    assert (waypoints[:, 0] == cfg.wm.history_size - 1).all()
    _assert_monotonic_and_valid(waypoints, seq_len=seq_len, min_stride=cfg.wm.high_level.waypoints.min_stride)

    spans = waypoints[:, -1] - waypoints[:, 0]
    assert (spans <= cfg.wm.high_level.waypoints.max_span).all()


def test_random_middle_requires_n3_and_warns():
    cfg = _make_cfg(
        strategy="random_middle",
        num_waypoints=5,
        min_stride=1,
        max_span=20,
    )
    with pytest.warns(UserWarning, match="defined for N=3 only"):
        with pytest.raises(ValueError, match="requires wm.high_level.waypoints.num=3"):
            SAMPLE_WAYPOINTS(cfg, batch_size=2, seq_len=30, device="cpu")


def test_random_sorted_sampling_constraints():
    torch.manual_seed(1)
    cfg = _make_cfg(
        strategy="random_sorted",
        num_waypoints=5,
        min_stride=1,
        max_span=15,
        history_size=3,
    )
    batch_size = 16
    seq_len = 50
    waypoints, gaps = SAMPLE_WAYPOINTS(cfg, batch_size=batch_size, seq_len=seq_len, device="cpu")

    assert waypoints.shape == (batch_size, 5)
    assert gaps.shape == (batch_size, 4)
    assert (waypoints[:, 0] == cfg.wm.history_size - 1).all()
    _assert_monotonic_and_valid(waypoints, seq_len=seq_len, min_stride=cfg.wm.high_level.waypoints.min_stride)

    spans = waypoints[:, -1] - waypoints[:, 0]
    min_total_span = (cfg.wm.high_level.waypoints.num - 1) * cfg.wm.high_level.waypoints.min_stride
    assert (spans >= min_total_span).all()
    assert (spans <= cfg.wm.high_level.waypoints.max_span).all()


def test_random_sorted_warns_if_stride_is_set():
    cfg = _make_cfg(
        strategy="random_sorted",
        num_waypoints=5,
        min_stride=1,
        max_span=15,
        stride=2,
    )
    with pytest.warns(UserWarning, match="stride=2 is ignored"):
        SAMPLE_WAYPOINTS(cfg, batch_size=2, seq_len=30, device="cpu")


def test_fixed_stride_sampling_exact_gaps():
    cfg = _make_cfg(
        strategy="fixed_stride",
        num_waypoints=6,
        min_stride=1,
        max_span=20,
        stride=2,
        history_size=3,
    )
    batch_size = 10
    seq_len = 40
    waypoints, gaps = SAMPLE_WAYPOINTS(cfg, batch_size=batch_size, seq_len=seq_len, device="cpu")

    assert waypoints.shape == (batch_size, 6)
    assert gaps.shape == (batch_size, 5)
    _assert_monotonic_and_valid(waypoints, seq_len=seq_len)
    assert (waypoints[:, 1:] - waypoints[:, :-1] == 2).all()


def test_fixed_stride_requires_positive_stride():
    cfg = _make_cfg(
        strategy="fixed_stride",
        num_waypoints=5,
        min_stride=1,
        max_span=20,
        stride=-1,
    )
    with pytest.raises(ValueError, match="stride must be > 0"):
        SAMPLE_WAYPOINTS(cfg, batch_size=2, seq_len=40, device="cpu")


def test_fixed_stride_warns_min_stride_ignored():
    cfg = _make_cfg(
        strategy="fixed_stride",
        num_waypoints=5,
        min_stride=10,
        max_span=20,
        stride=2,
    )
    with pytest.warns(UserWarning, match="min_stride=10 is ignored"):
        waypoints, _ = SAMPLE_WAYPOINTS(cfg, batch_size=2, seq_len=40, device="cpu")
    assert waypoints.shape == (2, 5)
