from __future__ import annotations

import warnings

import torch


def _sample_random_middle(
    *,
    batch_size: int,
    ctx_len: int,
    min_stride: int,
    max_available_span: int,
    device,
    beta_alpha: float,
    beta_beta: float,
):
    """Sample 3 sorted waypoints with a random middle split.

    Strategy:
        - Set first waypoint ``t1 = ctx_len - 1``.
        - Sample total span ``S`` uniformly in valid range.
        - Split ``S`` into two positive gaps ``gap1`` and ``gap2`` using a Beta draw.
        - Produce waypoints ``t1 < t2 < t3`` where:
            ``t2 = t1 + gap1``, ``t3 = t2 + gap2``.

    Args:
        batch_size: Number of sequences.
        ctx_len: Context length (history size).
        min_stride: Minimum gap between consecutive waypoints.
        max_available_span: Maximum allowed total span.
        device: Torch device.
        beta_alpha: Beta concentration parameter alpha.
        beta_beta: Beta concentration parameter beta.

    Returns:
        waypoints: ``(B, 3)``
        gaps: ``(B, 2)``
    """
    # This strategy is defined for N=3 only: t1 < t2 < t3.
    min_total_span = 2 * min_stride
    total_span = torch.randint(
        low=min_total_span,
        high=max_available_span + 1,
        size=(batch_size,),
        device=device,
    )

    beta_dist = torch.distributions.Beta(
        concentration1=torch.tensor(beta_alpha, device=device),
        concentration0=torch.tensor(beta_beta, device=device),
    )
    split_ratio = beta_dist.sample((batch_size,))
    gap1 = torch.round(split_ratio * total_span.float()).long()
    gap1 = torch.clamp(gap1, min=min_stride)
    gap1 = torch.minimum(gap1, total_span - min_stride)
    gap2 = total_span - gap1

    t1 = torch.full((batch_size,), ctx_len - 1, device=device, dtype=torch.long)
    t2 = t1 + gap1
    t3 = t2 + gap2
    waypoints = torch.stack([t1, t2, t3], dim=1)  # (B, 3)
    gaps = torch.stack([gap1, gap2], dim=1)  # (B, 2)
    return waypoints, gaps


def _sample_random_sorted(
    *,
    batch_size: int,
    num_waypoints: int,
    ctx_len: int,
    min_stride: int,
    max_available_span: int,
    device,
):
    """Sample sorted waypoints for arbitrary ``N`` via random gap allocation.

    Strategy:
        - Number of gaps is ``N-1``.
        - Each gap starts at ``min_stride``.
        - Sample a random total span ``S``.
        - Distribute residual span ``S - (N-1)*min_stride`` randomly across gaps.
        - Reconstruct waypoints by cumulative sum of gaps from ``t1 = ctx_len - 1``.

    This guarantees:
        - strict ordering ``t1 < t2 < ... < tN``
        - per-gap lower bound ``t_{k+1} - t_k >= min_stride``
        - sampled total span control

    Args:
        batch_size: Number of sequences.
        num_waypoints: Number of waypoints ``N``.
        ctx_len: Context length (history size).
        min_stride: Minimum gap between consecutive waypoints.
        max_available_span: Maximum allowed total span.
        device: Torch device.

    Returns:
        waypoints: ``(B, N)``
        gaps: ``(B, N-1)``
    """
    n_gaps = num_waypoints - 1
    min_total_span = n_gaps * min_stride
    total_span = torch.randint(
        low=min_total_span,
        high=max_available_span + 1,
        size=(batch_size,),
        device=device,
    )  # (B,)

    # Residual span distributed randomly across all gaps.
    residual = total_span - min_total_span  # (B,)
    probs = torch.full((batch_size, n_gaps), 1.0 / n_gaps, device=device)
    residual_alloc = torch.multinomial(
        probs,
        num_samples=int(residual.max().item()) if int(residual.max().item()) > 0 else 1,
        replacement=True,
    )  # (B, max_residual_or_1)

    gaps = torch.full((batch_size, n_gaps), min_stride, device=device, dtype=torch.long)
    if int(residual.max().item()) > 0:
        for b in range(batch_size):
            r = int(residual[b].item())
            if r > 0:
                idx = residual_alloc[b, :r]
                gaps[b].scatter_add_(
                    0,
                    idx,
                    torch.ones_like(idx, dtype=torch.long, device=device),
                )

    t1 = torch.full((batch_size, 1), ctx_len - 1, device=device, dtype=torch.long)
    waypoints = torch.cat([t1, t1 + gaps.cumsum(dim=1)], dim=1)  # (B, N)
    return waypoints, gaps


def _sample_fixed_stride(
    *,
    batch_size: int,
    num_waypoints: int,
    ctx_len: int,
    stride: int,
    device,
):
    """Sample sorted waypoints using a deterministic fixed stride.

    Waypoints are defined as:
        ``t1 = ctx_len - 1`` and ``t_{k+1} = t_k + stride``.

    Args:
        batch_size: Number of sequences.
        num_waypoints: Number of waypoints ``N``.
        ctx_len: Context length (history size).
        stride: Fixed waypoint gap.
        device: Torch device.

    Returns:
        waypoints: ``(B, N)``
        gaps: ``(B, N-1)`` all equal to ``stride``.
    """
    n_gaps = num_waypoints - 1
    gaps = torch.full((batch_size, n_gaps), stride, device=device, dtype=torch.long)
    t1 = torch.full((batch_size, 1), ctx_len - 1, device=device, dtype=torch.long)
    waypoints = torch.cat([t1, t1 + gaps.cumsum(dim=1)], dim=1)  # (B, N)
    return waypoints, gaps


def sample_waypoints(cfg, *, batch_size: int, seq_len: int, device):
    """Sample waypoint indices according to configured strategy and constraints.

    Supported strategies:
        - ``random_middle``: N=3 only
        - ``random_sorted``: generic N>=3, random non-fixed gaps
        - ``fixed_stride``: generic N>=3 with constant gap

    Global constraints enforced:
        - ``N >= 3``
        - ``min_stride > 0``
        - ``max_span > 0``
        - ``t1 = history_size - 1``
        - ``t_N - t_1 <= max_span``
        - last waypoint must remain inside sequence bounds

    Warning behavior:
        - ``stride`` is ignored unless strategy is ``fixed_stride``.
        - ``random_middle`` with ``N != 3`` warns then raises.

    Args:
        cfg: Hydra config.
        batch_size: Number of sequences.
        seq_len: Sequence length ``T``.
        device: Torch device.

    Returns:
        waypoints: ``(B, N)``
        gaps: ``(B, N-1)``
    """
    wcfg = cfg.wm.high_level.waypoints
    num_waypoints = int(wcfg.get("num", 5))
    if num_waypoints < 3:
        raise ValueError("wm.high_level.waypoints.num must be >= 3")

    strategy = str(wcfg.get("strategy", "random_sorted"))
    ctx_len = int(cfg.wm.history_size)
    min_stride = int(wcfg.get("min_stride", 1))
    max_span = int(wcfg.get("max_span", 1))
    stride_cfg = int(wcfg.get("stride", -1))
    if min_stride <= 0:
        raise ValueError("wm.high_level.waypoints.min_stride must be > 0")
    if max_span <= 0:
        raise ValueError("wm.high_level.waypoints.max_span must be > 0")

    max_available_span = min(max_span, seq_len - ctx_len)
    if max_available_span <= 0:
        raise ValueError(
            f"No valid future span available: max_available_span={max_available_span}. "
            "Increase dataset num_steps or reduce history_size/max_span."
        )

    if strategy != "fixed_stride" and stride_cfg != -1:
        warnings.warn(
            f"wm.high_level.waypoints.stride={stride_cfg} is ignored when "
            f"strategy={strategy}. Set stride=-1 to silence this warning.",
            stacklevel=2,
        )

    if strategy == "random_middle":
        if num_waypoints != 3:
            warnings.warn(
                "strategy=random_middle is defined for N=3 only. "
                f"Got wm.high_level.waypoints.num={num_waypoints}.",
                stacklevel=2,
            )
            raise ValueError("strategy=random_middle requires wm.high_level.waypoints.num=3")
        min_total_span = 2 * min_stride
        if max_available_span < min_total_span:
            raise ValueError(
                f"Sequence too short for random_middle: need span >= {min_total_span}, "
                f"got max_available_span={max_available_span}. "
                "Increase dataset num_steps or reduce min_stride/max_span."
            )
        beta_alpha = float(wcfg.get("beta_alpha", 2.0))
        beta_beta = float(wcfg.get("beta_beta", 2.0))
        if beta_alpha <= 0.0 or beta_beta <= 0.0:
            raise ValueError("waypoint beta_alpha and beta_beta must be > 0")
        waypoints, gaps = _sample_random_middle(
            batch_size=batch_size,
            ctx_len=ctx_len,
            min_stride=min_stride,
            max_available_span=max_available_span,
            device=device,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
        )
    elif strategy == "random_sorted":
        min_total_span = (num_waypoints - 1) * min_stride
        if max_available_span < min_total_span:
            raise ValueError(
                f"Sequence too short for random_sorted with N={num_waypoints}: "
                f"need span >= {min_total_span}, got max_available_span={max_available_span}. "
                "Increase dataset num_steps or reduce N/min_stride."
            )
        waypoints, gaps = _sample_random_sorted(
            batch_size=batch_size,
            num_waypoints=num_waypoints,
            ctx_len=ctx_len,
            min_stride=min_stride,
            max_available_span=max_available_span,
            device=device,
        )
    elif strategy == "fixed_stride":
        stride = stride_cfg
        if stride <= 0:
            raise ValueError(
                "wm.high_level.waypoints.stride must be > 0 when strategy=fixed_stride"
            )
        if min_stride != 1:
            warnings.warn(
                f"wm.high_level.waypoints.min_stride={min_stride} is ignored when "
                "strategy=fixed_stride. Set min_stride=1 to silence this warning.",
                stacklevel=2,
            )
        req_span = (num_waypoints - 1) * stride
        if req_span > max_available_span:
            raise ValueError(
                f"fixed_stride with N={num_waypoints}, stride={stride} requires span={req_span}, "
                f"but max_available_span={max_available_span}."
            )
        waypoints, gaps = _sample_fixed_stride(
            batch_size=batch_size,
            num_waypoints=num_waypoints,
            ctx_len=ctx_len,
            stride=stride,
            device=device,
        )
    else:
        raise ValueError(
            f"Unsupported waypoint strategy={strategy}. "
            "Use one of: random_middle, random_sorted, fixed_stride."
        )

    if (waypoints[:, -1] >= seq_len).any():
        raise RuntimeError("Waypoint sampling overflowed sequence bounds.")
    return waypoints, gaps
