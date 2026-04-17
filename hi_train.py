from __future__ import annotations

import re
import sys
import warnings
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from baseline_adapter import (
    ARPredictor,
    BASELINE_ROOT,
    Embedder,
    MLP,
    ModelObjectCallBack,
    SIGReg,
    get_column_normalizer,
    get_img_preprocessor,
)
from hi_jepa import HiJEPA
from hi_module import LatentActionEncoder


def _object_epoch(path: Path, source_policy: str) -> int | None:
    """Extract epoch index from an object-checkpoint filename.

    Expected filename pattern:
        <source_policy>_epoch_<N>_object.ckpt

    Args:
        path: Checkpoint path candidate.
        source_policy: Prefix expected in the filename.

    Returns:
        Parsed epoch integer if pattern matches, else ``None``.
    """
    match = re.match(rf"^{re.escape(source_policy)}_epoch_(\d+)_object\.ckpt$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def resolve_pretrained_checkpoint(cfg) -> Path:
    """Resolve pretrained low-level object checkpoint according to Hydra config.

    Supported selection modes:
        - ``explicit_path``: use ``pretrained_low_level.checkpoint.path`` directly
        - ``epoch``: pick ``<source_policy>_epoch_<epoch>_object.ckpt``
        - ``best``: prefer ``<source_policy>_best_object.ckpt`` if present,
          otherwise fallback to latest epoch checkpoint
        - ``latest``: pick highest epoch checkpoint

    Resolution root:
        - ``pretrained_low_level.checkpoint.search_dir`` if provided
        - otherwise StableWM cache dir

    Args:
        cfg: Full Hydra config.

    Returns:
        Concrete checkpoint path.

    Raises:
        ValueError: Invalid mode/config combination.
        FileNotFoundError: No valid checkpoint found.
    """
    pcfg = cfg.pretrained_low_level
    cpcfg = pcfg.checkpoint

    explicit = cpcfg.get("path")
    if explicit not in (None, ""):
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Explicit pretrained checkpoint does not exist: {path}")
        return path

    mode = str(cpcfg.get("selection_mode", "latest"))
    if mode == "explicit_path":
        raise ValueError(
            "pretrained_low_level.checkpoint.selection_mode=explicit_path requires "
            "pretrained_low_level.checkpoint.path to be set."
        )

    source_policy = str(pcfg.get("source_policy", "")).strip()
    if not source_policy:
        raise ValueError("pretrained_low_level.source_policy must be provided.")

    search_dir_raw = cpcfg.get("search_dir")
    search_dir = Path(search_dir_raw).expanduser() if search_dir_raw else Path(
        swm.data.utils.get_cache_dir()
    )
    if not search_dir.exists():
        raise FileNotFoundError(f"Checkpoint search_dir does not exist: {search_dir}")

    if mode == "epoch":
        epoch = int(cpcfg.get("epoch", 0))
        if epoch <= 0:
            raise ValueError("checkpoint.epoch must be > 0 when selection_mode=epoch")
        path = search_dir / f"{source_policy}_epoch_{epoch}_object.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"Epoch checkpoint not found: {path}")
        return path

    if mode not in {"latest", "best"}:
        raise ValueError(
            f"Unsupported pretrained checkpoint selection_mode={mode}. "
            "Use one of: latest, best, epoch, explicit_path."
        )

    if mode == "best":
        best_path = search_dir / f"{source_policy}_best_object.ckpt"
        if best_path.exists():
            return best_path

    candidates = []
    for path in search_dir.glob(f"{source_policy}_epoch_*_object.ckpt"):
        epoch = _object_epoch(path, source_policy)
        if epoch is not None:
            candidates.append((epoch, path))

    if not candidates:
        raise FileNotFoundError(
            f"No object checkpoints found for source_policy='{source_policy}' in {search_dir}"
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_pretrained_low_level_model(path: Path):
    """Load baseline LEWM object checkpoint and return the JEPA model object.

    This loader expects a serialized object checkpoint that either:
        - is already a model-like object, or
        - contains a ``model`` attribute (StablePretraining wrapper object).

    The returned model must expose the baseline LEWM modules:
        ``encoder``, ``predictor``, ``action_encoder``, ``projector``, ``pred_proj``.

    Args:
        path: Path to object checkpoint.

    Returns:
        Loaded model object with required LEWM components.

    Raises:
        ValueError: Checkpoint object does not match expected LEWM structure.
    """
    baseline_root = str(BASELINE_ROOT)
    if baseline_root not in sys.path:
        sys.path.insert(0, baseline_root)

    try:
        model_obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        model_obj = torch.load(path, map_location="cpu")

    model = model_obj.model if hasattr(model_obj, "model") else model_obj
    required = ("encoder", "predictor", "action_encoder", "projector", "pred_proj")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(
            f"Loaded object checkpoint does not look like LEWM JEPA model. Missing attrs: {missing}"
        )
    return model


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


def gather_waypoint_embeddings(emb: torch.Tensor, waypoints: torch.Tensor) -> torch.Tensor:
    """Gather latent embeddings at sampled waypoint indices.

    Args:
        emb: Full latent sequence, shape ``(B, T, D)``.
        waypoints: Waypoint indices per batch item, shape ``(B, N)``.

    Returns:
        Waypoint latents, shape ``(B, N, D)``.
    """
    if emb.ndim != 3:
        raise ValueError("emb must be shape (B, T, D)")
    if waypoints.ndim != 2:
        raise ValueError("waypoints must be shape (B, N)")
    b = emb.size(0)
    batch_idx = torch.arange(b, device=emb.device).unsqueeze(1)
    return emb[batch_idx, waypoints]


def build_action_chunks(
    actions: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded variable-length action chunks and validity mask.

    For each sample ``i`` in batch, this extracts:
        ``actions[i, starts[i]:ends[i], :]``
    and right-pads to the max chunk length in the batch.

    Args:
        actions: Primitive action sequence, shape ``(B, T, A)``.
        starts: Start indices per sample, shape ``(B,)``.
        ends: End indices per sample, shape ``(B,)``.

    Returns:
        chunks: Padded chunks, shape ``(B, T_chunk_max, A)``.
        mask: Valid-token mask, shape ``(B, T_chunk_max)`` where True is valid.
    """
    if actions.ndim != 3:
        raise ValueError("actions must be shape (B, T, A)")
    if starts.ndim != 1 or ends.ndim != 1:
        raise ValueError("starts/ends must be shape (B,)")
    if starts.shape != ends.shape:
        raise ValueError("starts and ends must have matching shape")

    lengths = (ends - starts).to(dtype=torch.long)
    if (lengths <= 0).any():
        raise ValueError("All action chunks must have positive length")

    b, _t, act_dim = actions.shape
    max_len = int(lengths.max().item())
    chunks = actions.new_zeros((b, max_len, act_dim))
    mask = torch.zeros((b, max_len), dtype=torch.bool, device=actions.device)

    for i in range(b):
        s = int(starts[i].item())
        e = int(ends[i].item())
        l = e - s
        chunks[i, :l] = actions[i, s:e]
        mask[i, :l] = True

    return chunks, mask


def hi_lejepa_forward(self, batch, stage, cfg):
    """Single train/val step for high-level predictor training.

    Pipeline:
        1. Encode observation sequence into latent states ``z``.
        2. Sample waypoints from configured strategy.
        3. Build action chunks between consecutive waypoints.
        4. Encode chunks into macro-actions via latent action encoder.
        5. Predict next waypoint latents using high-level predictor.
        6. Compute high-level prediction loss ``l2_pred_loss``.
        7. Optionally compute low-level loss (off by default).
        8. Optionally compute SIGReg (weight-controlled).
        9. Aggregate total loss and log metrics.

    Shapes:
        - emb: ``(B, T, D_z)``
        - waypoints: ``(B, N)``
        - z_context / z_target: ``(B, N-1, D_z)``
        - macro_actions: ``(B, N-1, D_l)``
        - z_pred: ``(B, N-1, D_z)``

    Args:
        self: StablePretraining module wrapper.
        batch: Batch dict from dataloader.
        stage: ``train`` or ``val``.
        cfg: Hydra config.

    Returns:
        Output dict containing losses and diagnostics.
    """
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    train_low_level = bool(cfg.training.get("train_low_level", False))
    output = self.model.encode(batch, encode_actions=train_low_level)

    emb = output["emb"]  # (B, T, D_z)
    actions = batch["action"]  # (B, T, A)
    b, t, _d = emb.shape
    device = emb.device

    waypoints, gaps = sample_waypoints(
        cfg,
        batch_size=b,
        seq_len=t,
        device=device,
    )
    z_waypoints = gather_waypoint_embeddings(emb, waypoints)  # (B, N, D_z)
    z_context = z_waypoints[:, :-1]  # (B, N-1, D_z)
    z_target = z_waypoints[:, 1:]  # (B, N-1, D_z)

    macro_actions_per_step = []
    macro_norms = []
    for k in range(z_context.size(1)):
        chunk_actions, chunk_mask = build_action_chunks(actions, waypoints[:, k], waypoints[:, k + 1])
        macro_k = self.model.encode_macro_actions(chunk_actions, chunk_mask)  # (B, D_l)
        macro_actions_per_step.append(macro_k)
        macro_norms.append(macro_k.norm(dim=-1))

    macro_actions = torch.stack(macro_actions_per_step, dim=1)  # (B, N-1, D_l)
    z_pred = self.model.predict_high(z_context, macro_actions)  # (B, N-1, D_z)
    output["l2_pred_loss"] = (z_pred - z_target).pow(2).mean()

    if train_low_level:
        ctx_len = int(cfg.wm.history_size)
        n_preds = int(cfg.wm.num_preds)
        act_emb = output["act_emb"]
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds : ctx_len + n_preds]
        pred_emb = self.model.predict_low(ctx_emb, ctx_act)  # (B, T_ctx, D_z)
        output["l1_pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    else:
        output["l1_pred_loss"] = torch.zeros((), device=device, dtype=emb.dtype)

    lambd = float(cfg.loss.sigreg.weight)
    if lambd > 0.0:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    else:
        output["sigreg_loss"] = torch.zeros((), device=device, dtype=emb.dtype)

    alpha = float(cfg.loss.get("alpha", 0.0))
    beta = float(cfg.loss.get("beta", 1.0))
    output["loss"] = (
        alpha * output["l1_pred_loss"]
        + beta * output["l2_pred_loss"]
        + lambd * output["sigreg_loss"]
    )

    output["waypoint_gap_mean"] = gaps.float().mean()
    output["waypoint_gap_max"] = gaps.float().max()
    output["macro_action_norm"] = torch.stack(macro_norms, dim=1).mean()

    metric_keys = (
        "loss",
        "l1_pred_loss",
        "l2_pred_loss",
        "sigreg_loss",
        "waypoint_gap_mean",
        "waypoint_gap_max",
        "macro_action_norm",
    )
    metrics = {f"{stage}/{k}": output[k].detach() for k in metric_keys}
    self.log_dict(metrics, on_step=True, sync_dist=True)
    return output


def summarize_params(module: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def log_param_breakdown(model: HiJEPA):
    """Print parameter distribution across core modules."""
    parts = [
        ("state_encoder", model.encoder),
        ("p1_low_predictor", model.low_predictor),
        ("p1_action_encoder", model.action_encoder),
        ("projector", model.projector),
        ("pred_proj", model.pred_proj),
        ("p2_high_predictor", model.high_predictor),
        ("p2_latent_action_encoder", model.latent_action_encoder),
        ("p2_macro_to_condition", model.macro_to_condition),
    ]

    total_all, trainable_all = summarize_params(model)
    print("[hi_train] parameter breakdown:")
    for name, module in parts:
        total, trainable = summarize_params(module)
        pct = (100.0 * total / total_all) if total_all > 0 else 0.0
        print(
            f"  - {name:24s} total={total:>12,} trainable={trainable:>12,} "
            f"share={pct:6.2f}%"
        )
    print(
        f"[hi_train] total params: {total_all:,} | trainable params: {trainable_all:,} "
        f"({(100.0 * trainable_all / total_all) if total_all > 0 else 0.0:.2f}% trainable)"
    )


def validate_high_level_config(cfg):
    """Validate high-level config consistency before model construction."""
    history_size = int(cfg.wm.history_size)
    num_steps = int(cfg.data.dataset.num_steps)
    max_span = int(cfg.wm.high_level.waypoints.max_span)
    max_seq_len = int(cfg.latent_action_encoder.max_seq_len)

    if num_steps <= history_size:
        raise ValueError(
            "data.dataset.num_steps must be > wm.history_size to allow future waypoint transitions."
        )

    max_available_span = min(max_span, num_steps - history_size)
    if max_available_span <= 0:
        raise ValueError(
            "No positive waypoint span available. Increase data.dataset.num_steps or reduce "
            "wm.high_level.waypoints.max_span / wm.history_size."
        )

    if max_seq_len < max_available_span:
        raise ValueError(
            "latent_action_encoder.max_seq_len is too small for waypoint sampling. "
            f"Need max_seq_len >= {max_available_span} (effective max span), "
            f"got {max_seq_len}. "
            "Set latent_action_encoder.max_seq_len to wm.high_level.waypoints.max_span "
            "or larger."
        )


@hydra.main(version_base=None, config_path="./config/train", config_name="hi_lewm")
def run(cfg):
    """Main training entrypoint for high-level predictor training.

    Responsibilities:
        - dataset/transforms setup
        - pretrained low-level checkpoint resolution/loading
        - model assembly (frozen low-level + trainable high-level path)
        - optimizer/scheduler wiring
        - trainer/manager launch

    Notes:
        - By default, encoder + low-level modules are frozen.
        - Default objective emphasizes high-level loss (``beta``) for PushT-focused runs.
    """
    validate_high_level_config(cfg)

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    effective_act_dim = int(cfg.data.dataset.frameskip) * int(cfg.wm.action_dim)

    if bool(cfg.pretrained_low_level.enabled):
        ckpt_path = resolve_pretrained_checkpoint(cfg)
        pretrained = load_pretrained_low_level_model(ckpt_path)
        print(f"[hi_train] loaded pretrained low-level object: {ckpt_path}")

        encoder = pretrained.encoder
        low_predictor = pretrained.predictor
        action_encoder = pretrained.action_encoder
        projector = pretrained.projector
        predictor_proj = pretrained.pred_proj
    else:
        encoder = spt.backbone.utils.vit_hf(
            cfg.encoder_scale,
            patch_size=cfg.patch_size,
            image_size=cfg.img_size,
            pretrained=False,
            use_mask_token=False,
        )
        hidden_dim = encoder.config.hidden_size
        embed_dim = int(cfg.wm.get("embed_dim", hidden_dim))
        low_predictor = ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        )
        action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
        projector = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )
        predictor_proj = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

    if hasattr(low_predictor, "pos_embedding"):
        embed_dim = int(low_predictor.pos_embedding.shape[-1])
    else:
        embed_dim = int(cfg.wm.get("embed_dim", 192))

    if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
        hidden_dim = int(encoder.config.hidden_size)
    else:
        hidden_dim = embed_dim

    num_waypoints = int(cfg.wm.high_level.waypoints.num)
    if num_waypoints < 3:
        raise ValueError("wm.high_level.waypoints.num must be >= 3")
    high_num_frames = num_waypoints - 1

    high_pred_cfg = dict(cfg.predictor_high)
    high_predictor = ARPredictor(
        num_frames=high_num_frames,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **high_pred_cfg,
    )

    latent_action_dim = int(cfg.wm.high_level.get("latent_action_dim", embed_dim))
    latent_encoder_cfg = dict(cfg.latent_action_encoder)
    latent_action_encoder = LatentActionEncoder(
        input_dim=effective_act_dim,
        latent_dim=latent_action_dim,
        **latent_encoder_cfg,
    )

    cond_dim = embed_dim
    proj_mode = str(cfg.wm.high_level.get("macro_to_condition_proj", "auto"))
    if proj_mode == "identity":
        if latent_action_dim != cond_dim:
            raise ValueError(
                "macro_to_condition_proj=identity requires "
                "latent_action_dim == wm.embed_dim"
            )
        macro_to_condition = torch.nn.Identity()
    elif proj_mode == "linear":
        macro_to_condition = torch.nn.Linear(latent_action_dim, cond_dim)
    elif proj_mode == "auto":
        macro_to_condition = (
            torch.nn.Identity()
            if latent_action_dim == cond_dim
            else torch.nn.Linear(latent_action_dim, cond_dim)
        )
    else:
        raise ValueError(
            f"Unsupported wm.high_level.macro_to_condition_proj={proj_mode}. "
            "Use one of: auto, identity, linear."
        )

    world_model = HiJEPA(
        encoder=encoder,
        low_predictor=low_predictor,
        action_encoder=action_encoder,
        high_predictor=high_predictor,
        latent_action_encoder=latent_action_encoder,
        macro_to_condition=macro_to_condition,
        projector=projector,
        pred_proj=predictor_proj,
    )

    freeze_cfg = cfg.pretrained_low_level.freeze
    freeze_encoder = bool(freeze_cfg.get("encoder", True))
    freeze_low_predictor = bool(freeze_cfg.get("low_level_predictor", True))
    freeze_action_encoder = bool(freeze_cfg.get("low_level_action_encoder", True))
    freeze_projector = bool(freeze_cfg.get("projector", True))
    freeze_pred_proj = bool(freeze_cfg.get("pred_proj", True))

    if bool(cfg.pretrained_low_level.enabled):
        world_model.freeze_low_level(
            freeze_encoder=freeze_encoder,
            freeze_low_predictor=freeze_low_predictor,
            freeze_action_encoder=freeze_action_encoder,
            freeze_projector=freeze_projector,
            freeze_pred_proj=freeze_pred_proj,
        )
    else:
        if any(
            (
                freeze_encoder,
                freeze_low_predictor,
                freeze_action_encoder,
                freeze_projector,
                freeze_pred_proj,
            )
        ):
            warnings.warn(
                "pretrained_low_level.enabled=False, so pretrained freeze settings are ignored. "
                "Low-level modules remain trainable.",
                stacklevel=2,
            )
        world_model.freeze_low_level(
            freeze_encoder=False,
            freeze_low_predictor=False,
            freeze_action_encoder=False,
            freeze_projector=False,
            freeze_pred_proj=False,
        )

    log_param_breakdown(world_model)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(hi_lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        wandb_cfg = OmegaConf.to_container(cfg.wandb.config, resolve=True)
        if wandb_cfg.get("entity") in (None, ""):
            wandb_cfg.pop("entity", None)
        logger = WandbLogger(**wandb_cfg)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=int(cfg.checkpointing.object_dump.epoch_interval),
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()


if __name__ == "__main__":
    run()
