from __future__ import annotations

import re
import sys
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torch.utils.data import default_collate

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
from hi_waypoint_sampling import sample_waypoints


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


def build_action_chunks_batched(
    actions: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded action chunks for all waypoint transitions in one pass.

    Args:
        actions: Primitive action sequence, shape ``(B, T, A)``.
        starts: Start indices per sample/transition, shape ``(B, K)``.
        ends: End indices per sample/transition, shape ``(B, K)``.

    Returns:
        chunks: Padded chunks, shape ``(B, K, L_max, A)``.
        mask: Valid-token mask, shape ``(B, K, L_max)`` where True is valid.
    """
    if actions.ndim != 3:
        raise ValueError("actions must be shape (B, T, A)")
    if starts.ndim != 2 or ends.ndim != 2:
        raise ValueError("starts/ends must be shape (B, K)")
    if starts.shape != ends.shape:
        raise ValueError("starts and ends must have matching shape")

    b, t, act_dim = actions.shape
    if starts.size(0) != b:
        raise ValueError("starts/ends batch dimension must match actions")
    if (starts < 0).any() or (ends < 0).any() or (starts >= t).any() or (ends > t).any():
        raise ValueError("starts/ends must satisfy 0 <= starts < T and 0 <= ends <= T")

    lengths = (ends - starts).to(dtype=torch.long)
    if (lengths <= 0).any():
        raise ValueError("All action chunks must have positive length")

    max_len = int(lengths.max().item())
    offsets = torch.arange(max_len, device=actions.device).view(1, 1, max_len)
    mask = offsets < lengths.unsqueeze(-1)  # (B, K, L_max)

    gather_idx = starts.unsqueeze(-1) + offsets
    gather_idx = gather_idx.clamp(min=0, max=t - 1)

    batch_idx = torch.arange(b, device=actions.device).view(b, 1, 1)
    batch_idx = batch_idx.expand_as(gather_idx)
    chunks = actions[batch_idx, gather_idx, :]  # (B, K, L_max, A)
    chunks = chunks * mask.unsqueeze(-1).to(dtype=actions.dtype)

    if chunks.shape[-1] != act_dim:
        raise RuntimeError("Unexpected chunk action dimension mismatch")
    return chunks, mask


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


def build_p2_frozen_waypoint_collate(cfg, pixel_preprocessor):
    """Build collate_fn that preprocesses only sampled waypoint pixel frames."""
    if pixel_preprocessor is None:
        raise ValueError("pixel_preprocessor is required for P2 frozen waypoint collate.")

    num_waypoints = int(cfg.wm.high_level.waypoints.num)

    def collate(samples):
        if len(samples) == 0:
            raise ValueError("Cannot collate an empty batch.")

        seq_len = int(samples[0]["action"].shape[0])
        waypoints, _ = sample_waypoints(
            cfg,
            batch_size=len(samples),
            seq_len=seq_len,
            device="cpu",
        )

        processed = []
        for i, sample in enumerate(samples):
            item = dict(sample)
            wp = waypoints[i]
            pixels = item["pixels"]
            if torch.is_tensor(pixels):
                selected = pixels.index_select(
                    0,
                    wp.to(device=pixels.device, dtype=torch.long),
                )
            else:
                selected = pixels[wp.cpu().numpy()]

            pixel_out = pixel_preprocessor({"pixels": selected})
            item["pixels"] = pixel_out["pixels"]
            item["waypoints"] = wp.clone()
            processed.append(item)

        batch = default_collate(processed)
        if batch["pixels"].shape[1] != num_waypoints:
            raise RuntimeError(
                f"Expected waypoint pixel length={num_waypoints}, got {batch['pixels'].shape[1]}"
            )
        return batch

    return collate


def hi_lejepa_forward(self, batch, stage, cfg):
    """Single train/val step for high-level predictor training.

    Pipeline:
        1. Sample waypoints from configured strategy.
        2. Encode either:
           - full observation sequence, or
           - waypoint-only frames in P2 fast path.
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
    actions = batch["action"]  # (B, T, A)
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

    # P2 fast path: avoid encoding all T frames when only waypoint latents are needed.
    use_waypoint_fast_path = (not train_low_level) and (lambd <= 0.0)
    if use_waypoint_fast_path:
        output = {}
        emb = None
        z_waypoints = self.model.encode_selected_frames(batch["pixels"], waypoints)
    else:
        output = self.model.encode(batch, encode_actions=train_low_level)
        emb = output["emb"]  # (B, T, D_z)
        z_waypoints = gather_waypoint_embeddings(emb, waypoints)  # (B, N, D_z)

    z_context = z_waypoints[:, :-1]  # (B, N-1, D_z)
    z_target = z_waypoints[:, 1:]  # (B, N-1, D_z)

    starts = waypoints[:, :-1]
    ends = waypoints[:, 1:]
    chunk_actions, chunk_mask = build_action_chunks_batched(actions, starts, ends)
    _, k, l_max, act_dim = chunk_actions.shape
    flat_actions = chunk_actions.reshape(b * k, l_max, act_dim)
    flat_mask = chunk_mask.reshape(b * k, l_max)
    flat_macro = self.model.encode_macro_actions(flat_actions, flat_mask)  # (B*K, D_l)
    macro_actions = flat_macro.reshape(b, k, -1)  # (B, N-1, D_l)

    z_pred = self.model.predict_high(z_context, macro_actions)  # (B, N-1, D_z)
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
        pred_emb = self.model.predict_low(ctx_emb, ctx_act)  # (B, T_ctx, D_z)
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
    output["loss"] = (
        alpha * output["l1_pred_loss"]
        + beta * output["l2_pred_loss"]
        + lambd * output["sigreg_loss"]
    )

    output["waypoint_gap_mean"] = gaps.float().mean()
    output["waypoint_gap_max"] = gaps.float().max()
    output["macro_action_norm"] = macro_actions.norm(dim=-1).mean()

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


def hi_lejepa_forward_p2_frozen(self, batch, stage, cfg):
    """Train/val step for P2-only runs with frozen low-level modules.

    Assumes:
        - ``batch['pixels']`` already contains only sampled waypoint frames ``(B, N, C, H, W)``
        - ``batch['waypoints']`` contains original sequence indices ``(B, N)``
        - low-level loss and SIGReg are disabled
    """
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    actions = batch["action"]  # (B, T, A)
    b, _t, _a = actions.shape
    device = actions.device

    if "waypoints" not in batch:
        raise RuntimeError("P2-frozen forward requires precomputed `batch['waypoints']`.")
    waypoints = batch["waypoints"].to(device=device, dtype=torch.long)
    if waypoints.ndim != 2 or waypoints.size(0) != b:
        raise ValueError("waypoints must be shape (B, N) and match action batch size.")

    output = self.model.encode({"pixels": batch["pixels"]}, encode_actions=False)
    z_waypoints = output["emb"]  # (B, N, D_z)
    if z_waypoints.size(1) != waypoints.size(1):
        raise RuntimeError("Waypoint pixel count and waypoint index count do not match.")

    z_context = z_waypoints[:, :-1]  # (B, N-1, D_z)
    z_target = z_waypoints[:, 1:]  # (B, N-1, D_z)

    starts = waypoints[:, :-1]
    ends = waypoints[:, 1:]
    chunk_actions, chunk_mask = build_action_chunks_batched(actions, starts, ends)
    _, k, l_max, act_dim = chunk_actions.shape
    flat_actions = chunk_actions.reshape(b * k, l_max, act_dim)
    flat_mask = chunk_mask.reshape(b * k, l_max)
    flat_macro = self.model.encode_macro_actions(flat_actions, flat_mask)  # (B*K, D_l)
    macro_actions = flat_macro.reshape(b, k, -1)  # (B, N-1, D_l)

    z_pred = self.model.predict_high(z_context, macro_actions)  # (B, N-1, D_z)
    output["l2_pred_loss"] = (z_pred - z_target).pow(2).mean()
    output["l1_pred_loss"] = torch.zeros((), device=device, dtype=z_waypoints.dtype)
    output["sigreg_loss"] = torch.zeros((), device=device, dtype=z_waypoints.dtype)

    alpha = float(cfg.loss.get("alpha", 0.0))
    beta = float(cfg.loss.get("beta", 1.0))
    output["loss"] = alpha * output["l1_pred_loss"] + beta * output["l2_pred_loss"]

    gaps = waypoints[:, 1:] - waypoints[:, :-1]
    output["waypoint_gap_mean"] = gaps.float().mean()
    output["waypoint_gap_max"] = gaps.float().max()
    output["macro_action_norm"] = macro_actions.norm(dim=-1).mean()

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


def clone_projection_head(module: torch.nn.Module) -> torch.nn.Module:
    """Return a trainable deep copy of a projection head module."""
    return deepcopy(module)


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
        ("p1_low_pred_proj", model.low_pred_proj),
        ("p2_high_pred_proj", model.high_pred_proj),
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

    use_p2_frozen_optimization = is_p2_frozen_optimization_enabled(cfg)
    if use_p2_frozen_optimization:
        print("[hi_train] enabling P2 frozen input optimization (waypoint-only pixel preprocessing).")

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    pixel_preprocessor = None
    transforms = []
    if use_p2_frozen_optimization:
        pixel_preprocessor = get_img_preprocessor(
            source="pixels",
            target="pixels",
            img_size=cfg.img_size,
        )
    else:
        transforms.append(get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size))

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms) if transforms else None
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    loader_kwargs = dict(cfg.loader)
    if use_p2_frozen_optimization:
        loader_kwargs["collate_fn"] = build_p2_frozen_waypoint_collate(cfg, pixel_preprocessor)

    train = torch.utils.data.DataLoader(
        train_set, **loader_kwargs, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **loader_kwargs, shuffle=False, drop_last=False
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
        low_predictor_proj = pretrained.pred_proj
        high_predictor_proj = clone_projection_head(pretrained.pred_proj)
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
        low_predictor_proj = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )
        high_predictor_proj = MLP(
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
        low_pred_proj=low_predictor_proj,
        high_pred_proj=high_predictor_proj,
    )

    freeze_cfg = cfg.pretrained_low_level.freeze
    freeze_encoder = bool(freeze_cfg.get("encoder", True))
    freeze_low_predictor = bool(freeze_cfg.get("low_level_predictor", True))
    freeze_action_encoder = bool(freeze_cfg.get("low_level_action_encoder", True))
    freeze_projector = bool(freeze_cfg.get("projector", True))
    freeze_low_pred_proj = bool(freeze_cfg.get("low_pred_proj", True))
    freeze_high_pred_proj = bool(freeze_cfg.get("high_pred_proj", False))

    if bool(cfg.pretrained_low_level.enabled):
        world_model.freeze_low_level(
            freeze_encoder=freeze_encoder,
            freeze_low_predictor=freeze_low_predictor,
            freeze_action_encoder=freeze_action_encoder,
            freeze_projector=freeze_projector,
            freeze_low_pred_proj=freeze_low_pred_proj,
            freeze_high_pred_proj=freeze_high_pred_proj,
        )
    else:
        if any(
            (
                freeze_encoder,
                freeze_low_predictor,
                freeze_action_encoder,
                freeze_projector,
                freeze_low_pred_proj,
                freeze_high_pred_proj,
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
            freeze_low_pred_proj=False,
            freeze_high_pred_proj=False,
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

    selected_forward = (
        hi_lejepa_forward_p2_frozen if use_p2_frozen_optimization else hi_lejepa_forward
    )

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(selected_forward, cfg=cfg),
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
