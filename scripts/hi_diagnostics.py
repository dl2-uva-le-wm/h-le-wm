#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h_le_wm.baseline.adapter as _baseline_adapter

# Backward-compatibility for torch.load on object checkpoints saved by hi_train.
_ = _baseline_adapter.ARPredictor

DEFAULT_SUBGOAL_OFFSETS = (2, 3, 5)


@dataclass(slots=True)
class DiagnosticConfig:
    policy: str
    experiment_kind: str
    dataset_name: str = "pusht_expert_train"
    cache_dir: str | None = None
    img_size: int = 224
    num_eval_samples: int = 256
    num_empirical_chunks: int = 4096
    goal_offset_steps: int = 25
    high_horizon: int = 1
    low_horizon: int = 2
    high_num_samples: int = 900
    high_iters: int = 20
    high_topk: int = 10
    low_num_samples: int = 300
    low_iters: int = 30
    low_topk: int = 150
    cem_elite_frac: float = 0.1
    cem_bound_mode: str = "none"
    frame_skip: int = 5
    seed: int = 42
    device: str = "auto"
    subgoal_offsets: tuple[int, ...] = DEFAULT_SUBGOAL_OFFSETS
    reference_latent_pool_size: int = 4096
    max_cov_dim_for_json: int = 64
    save_json: str | None = None
    save_npz: str | None = None
    append_tsv: str | None = None


@dataclass(slots=True)
class ReferenceStats:
    span_tokens: int
    samples: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    q01: torch.Tensor
    q05: torch.Tensor
    q95: torch.Tensor
    q99: torch.Tensor
    cov: torch.Tensor
    inv_cov: torch.Tensor
    md2: torch.Tensor


@dataclass(slots=True)
class DiagnosticContext:
    cfg: DiagnosticConfig
    device: torch.device
    rng: np.random.Generator
    dataset: Any
    model: torch.nn.Module
    action: np.ndarray
    episode_ids: np.ndarray
    step_idx: np.ndarray | None
    action_scaler: preprocessing.StandardScaler
    tfm: Any
    latent_dim: int
    raw_action_dim: int
    macro_input_dim: int
    group: int
    low_action_input_dim: int
    reference_cache: dict[int, ReferenceStats]
    action_token_reference_cache: dict[str, dict[str, torch.Tensor]]
    latent_pool_cache: dict[int, tuple[np.ndarray, torch.Tensor]]


def img_transform(img_size: int):
    imagenet_stats = getattr(spt.data, "dataset_stats", None)
    if imagenet_stats is not None and hasattr(imagenet_stats, "ImageNet"):
        norm_kwargs = imagenet_stats.ImageNet
    else:
        norm_kwargs = {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**norm_kwargs),
            transforms.Resize(size=img_size),
        ]
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def infer_latent_dim(model: torch.nn.Module) -> int:
    if hasattr(model, "_infer_latent_action_dim"):
        return int(model._infer_latent_action_dim())  # type: ignore[attr-defined]
    if hasattr(model, "latent_action_encoder"):
        enc = model.latent_action_encoder
        if hasattr(enc, "output_proj") and hasattr(enc.output_proj, "out_features"):
            return int(enc.output_proj.out_features)
        if hasattr(enc, "latent_dim"):
            return int(enc.latent_dim)
    raise ValueError("Unable to infer latent action dimension.")


def infer_macro_input_dim(model: torch.nn.Module) -> int | None:
    if not hasattr(model, "latent_action_encoder"):
        return None
    enc = model.latent_action_encoder
    if hasattr(enc, "input_proj") and hasattr(enc.input_proj, "in_features"):
        return int(enc.input_proj.in_features)
    return None


def infer_low_action_input_dim(model: torch.nn.Module) -> int:
    if not hasattr(model, "action_encoder"):
        raise ValueError("Model does not expose action_encoder.")
    enc = model.action_encoder
    if hasattr(enc, "patch_embed") and hasattr(enc.patch_embed, "in_channels"):
        return int(enc.patch_embed.in_channels)
    raise ValueError("Unable to infer low-level action input dimension from action_encoder.")


def get_episode_col_name(dataset) -> str:
    if "episode_idx" in dataset.column_names:
        return "episode_idx"
    if "ep_idx" in dataset.column_names:
        return "ep_idx"
    raise ValueError("Dataset has neither 'episode_idx' nor 'ep_idx'.")


def contiguous_valid_starts(
    *,
    episode_ids: np.ndarray,
    step_idx: np.ndarray | None,
    seq_len: int,
    chunk_len: int,
) -> np.ndarray:
    if chunk_len <= 0 or seq_len < chunk_len:
        return np.empty((0,), dtype=np.int64)

    max_start = seq_len - chunk_len
    if episode_ids is None:
        return np.arange(max_start + 1, dtype=np.int64)

    ep_change = episode_ids[1:] != episode_ids[:-1]
    bad_transition = ep_change.copy()
    if step_idx is not None:
        bad_transition |= (step_idx[1:] - step_idx[:-1]) != 1

    transitions_per_chunk = chunk_len - 1
    if transitions_per_chunk <= 0:
        return np.arange(max_start + 1, dtype=np.int64)

    bad_i64 = bad_transition.astype(np.int64)
    csum = np.cumsum(np.concatenate(([0], bad_i64)))
    window_bad = csum[transitions_per_chunk:] - csum[:-transitions_per_chunk]
    return np.nonzero(window_bad == 0)[0].astype(np.int64)


def sample_indices(rng: np.random.Generator, pool: np.ndarray, n: int) -> np.ndarray:
    if pool.size == 0:
        return pool
    replace = pool.size < n
    pick = rng.choice(pool.size, size=n, replace=replace)
    return np.asarray(pool[pick], dtype=np.int64)


def get_row_data_safe(dataset, row_idx: np.ndarray) -> dict[str, np.ndarray]:
    row_idx = np.asarray(row_idx, dtype=np.int64)
    order = np.argsort(row_idx, kind="mergesort")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    sorted_idx = row_idx[order]
    rows_sorted = dataset.get_row_data(sorted_idx)
    return {k: np.asarray(v)[inv_order] for k, v in rows_sorted.items()}


def preprocess_actions(
    actions_raw: np.ndarray,
    action_scaler: preprocessing.StandardScaler,
    *,
    chunk_len_tokens: int,
    group: int,
) -> np.ndarray:
    b, raw_chunk_len, raw_dim = actions_raw.shape
    flat = actions_raw.reshape(-1, raw_dim)
    norm = action_scaler.transform(flat).reshape(b, raw_chunk_len, raw_dim)
    if group == 1:
        return norm
    return norm.reshape(b, chunk_len_tokens, raw_dim * group)


def encode_pixels_last(model: torch.nn.Module, pixels_bhwc: np.ndarray, tfm, device: torch.device) -> torch.Tensor:
    pixels_chw = torch.stack([tfm(x) for x in pixels_bhwc], dim=0).to(device)
    batch = {"pixels": pixels_chw.unsqueeze(1)}
    out = model.encode(batch, encode_actions=False)
    return out["emb"][:, -1]


def encode_macro_actions(
    model: torch.nn.Module,
    actions_tokens: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    a = torch.from_numpy(actions_tokens.astype(np.float32)).to(device)
    mask = torch.ones((a.size(0), a.size(1)), dtype=torch.bool, device=device)
    return model.encode_macro_actions(a, mask)


def rollout_one_high(model: torch.nn.Module, z_init: torch.Tensor, macro_actions: torch.Tensor) -> torch.Tensor:
    pred = model.rollout_high(z_init, macro_actions.unsqueeze(1))
    return pred[:, 0, 0, :]


def mahalanobis_sq(x: torch.Tensor, mean: torch.Tensor, inv_cov: torch.Tensor) -> torch.Tensor:
    diff = x - mean
    return torch.einsum("...d,dd,...d->...", diff, inv_cov, diff)


def build_action_scaler(dataset) -> preprocessing.StandardScaler:
    action = np.asarray(dataset.get_col_data("action"))
    valid = ~np.isnan(action).any(axis=1)
    scaler = preprocessing.StandardScaler()
    scaler.fit(action[valid])
    return scaler


def resolve_cache_dir(cache_dir: str | None) -> Path:
    candidates: list[Path] = []
    if cache_dir:
        candidates.append(Path(cache_dir))
    else:
        for env_key in ("STABLEWM_DIAGNOSTICS_CACHE_DIR", "STABLEWM_CACHE_DIR"):
            raw = os.environ.get(env_key)
            if raw:
                candidates.append(Path(raw))
        stablewm_home = os.environ.get("STABLEWM_HOME")
        if stablewm_home:
            candidates.append(Path(stablewm_home))
            candidates.append(Path(stablewm_home) / "cache" / "stable_worldmodel")
        candidates.append(Path(tempfile.gettempdir()) / f"stable_worldmodel_{os.environ.get('USER', 'user')}")
        candidates.append(Path.cwd() / ".stable_worldmodel_cache")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    raise OSError("Could not create a writable cache directory for stable_worldmodel data access.")


def build_context(cfg: DiagnosticConfig) -> DiagnosticContext:
    device = resolve_device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    cache_dir = resolve_cache_dir(cfg.cache_dir)
    dataset = swm.data.HDF5Dataset(
        cfg.dataset_name,
        keys_to_cache=["action"],
        cache_dir=cache_dir,
    )

    episode_col = get_episode_col_name(dataset)
    episode_ids = np.asarray(dataset.get_col_data(episode_col))
    step_idx = np.asarray(dataset.get_col_data("step_idx")) if "step_idx" in dataset.column_names else None
    action = np.asarray(dataset.get_col_data("action"))
    raw_action_dim = int(action.shape[1])

    model = swm.policy.AutoCostModel(cfg.policy, cache_dir=cache_dir).to(device).eval()
    model.requires_grad_(False)

    latent_dim = infer_latent_dim(model)
    macro_input_dim = infer_macro_input_dim(model)
    if macro_input_dim is None:
        macro_input_dim = raw_action_dim
    if macro_input_dim % raw_action_dim != 0:
        raise ValueError(
            f"Model macro input dim {macro_input_dim} is not divisible by dataset action dim {raw_action_dim}."
        )
    group = macro_input_dim // raw_action_dim
    low_action_input_dim = infer_low_action_input_dim(model)
    action_scaler = build_action_scaler(dataset)
    tfm = img_transform(cfg.img_size)

    return DiagnosticContext(
        cfg=cfg,
        device=device,
        rng=rng,
        dataset=dataset,
        model=model,
        action=action,
        episode_ids=episode_ids,
        step_idx=step_idx,
        action_scaler=action_scaler,
        tfm=tfm,
        latent_dim=latent_dim,
        raw_action_dim=raw_action_dim,
        macro_input_dim=macro_input_dim,
        group=group,
        low_action_input_dim=low_action_input_dim,
        reference_cache={},
        action_token_reference_cache={},
        latent_pool_cache={},
    )


def goal_tokens(cfg: DiagnosticConfig) -> int:
    return max(1, int(math.ceil(int(cfg.goal_offset_steps) / int(cfg.frame_skip))))


def parse_policy_run_info(policy: str) -> tuple[str, int | None]:
    name = Path(policy).name
    if "_epoch_" in name:
        run_name, epoch_raw = name.rsplit("_epoch_", 1)
        if epoch_raw.isdigit():
            return run_name, int(epoch_raw)
    return name, None


def partition_total(total: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be > 0")
    base = total // parts
    rem = total % parts
    out = []
    for idx in range(parts):
        out.append(base + (1 if idx >= parts - rem and rem > 0 else 0))
    if any(x <= 0 for x in out):
        raise ValueError(f"Cannot partition total={total} into {parts} positive chunks.")
    return out


def sample_valid_starts_for_raw_span(ctx: DiagnosticContext, raw_span: int, n: int) -> np.ndarray:
    valid_starts = contiguous_valid_starts(
        episode_ids=ctx.episode_ids,
        step_idx=ctx.step_idx,
        seq_len=int(ctx.action.shape[0]),
        chunk_len=raw_span + 1,
    )
    if valid_starts.size == 0:
        raise ValueError(f"No valid contiguous starts found for raw_span={raw_span}.")
    return sample_indices(ctx.rng, valid_starts, n)


def encode_row_latents(ctx: DiagnosticContext, row_idx: np.ndarray) -> torch.Tensor:
    rows = get_row_data_safe(ctx.dataset, row_idx)
    pixels = np.asarray(rows["pixels"])
    return encode_pixels_last(ctx.model, pixels, ctx.tfm, ctx.device)


def build_macro_reference(ctx: DiagnosticContext, span_tokens: int) -> ReferenceStats:
    if span_tokens in ctx.reference_cache:
        return ctx.reference_cache[span_tokens]

    raw_span = int(span_tokens) * ctx.group
    starts = sample_valid_starts_for_raw_span(ctx, raw_span=raw_span, n=ctx.cfg.num_empirical_chunks)
    chunks_raw = np.stack([ctx.action[s : s + raw_span] for s in starts], axis=0)
    chunks_tok = preprocess_actions(
        chunks_raw,
        ctx.action_scaler,
        chunk_len_tokens=int(span_tokens),
        group=ctx.group,
    )
    samples = encode_macro_actions(ctx.model, chunks_tok, device=ctx.device)
    ref = build_reference_stats(samples, span_tokens=int(span_tokens))
    ctx.reference_cache[span_tokens] = ref
    return ref


def build_reference_stats(samples: torch.Tensor, span_tokens: int) -> ReferenceStats:
    mean = samples.mean(dim=0)
    std = samples.std(dim=0, unbiased=False).clamp_min(1e-3)
    q01 = torch.quantile(samples, 0.01, dim=0)
    q05 = torch.quantile(samples, 0.05, dim=0)
    q95 = torch.quantile(samples, 0.95, dim=0)
    q99 = torch.quantile(samples, 0.99, dim=0)
    centered = samples - mean
    cov = (centered.T @ centered) / max(1, int(samples.size(0)) - 1)
    cov = cov + 1e-4 * torch.eye(int(samples.size(1)), device=samples.device, dtype=cov.dtype)
    inv_cov = torch.linalg.pinv(cov)
    md2 = mahalanobis_sq(samples, mean, inv_cov)
    ref = ReferenceStats(
        span_tokens=int(span_tokens),
        samples=samples,
        mean=mean,
        std=std,
        q01=q01,
        q05=q05,
        q95=q95,
        q99=q99,
        cov=cov,
        inv_cov=inv_cov,
        md2=md2,
    )
    return ref


def build_action_token_reference(ctx: DiagnosticContext) -> dict[str, torch.Tensor]:
    cache_key = f"group={ctx.group}"
    if cache_key in ctx.action_token_reference_cache:
        return ctx.action_token_reference_cache[cache_key]

    starts = sample_valid_starts_for_raw_span(ctx, raw_span=ctx.group, n=ctx.cfg.num_empirical_chunks)
    chunks_raw = np.stack([ctx.action[s : s + ctx.group] for s in starts], axis=0)
    chunks_tok = preprocess_actions(
        chunks_raw,
        ctx.action_scaler,
        chunk_len_tokens=1,
        group=ctx.group,
    ).reshape(len(starts), ctx.low_action_input_dim)
    token_t = torch.from_numpy(chunks_tok.astype(np.float32)).to(ctx.device)
    mean = token_t.mean(dim=0)
    std = token_t.std(dim=0, unbiased=False).clamp_min(1e-3)
    q05 = torch.quantile(token_t, 0.05, dim=0)
    q95 = torch.quantile(token_t, 0.95, dim=0)
    out = {"samples": token_t, "mean": mean, "std": std, "q05": q05, "q95": q95}
    ctx.action_token_reference_cache[cache_key] = out
    return out


def get_reference_latent_pool(ctx: DiagnosticContext, n_rows: int) -> tuple[np.ndarray, torch.Tensor]:
    if n_rows in ctx.latent_pool_cache:
        return ctx.latent_pool_cache[n_rows]
    pool = np.arange(int(ctx.action.shape[0]), dtype=np.int64)
    rows = sample_indices(ctx.rng, pool, n_rows)
    latents = encode_row_latents(ctx, rows)
    ctx.latent_pool_cache[n_rows] = (rows, latents)
    return rows, latents


def expand_bounds(refs: Sequence[ReferenceStats], mode: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if mode == "none":
        return None, None
    if mode not in {"q01_q99", "q05_q95"}:
        raise ValueError(f"Unsupported cem_bound_mode={mode}")
    low_attr = "q01" if mode == "q01_q99" else "q05"
    high_attr = "q99" if mode == "q01_q99" else "q95"
    low = torch.stack([getattr(ref, low_attr) for ref in refs], dim=0)
    high = torch.stack([getattr(ref, high_attr) for ref in refs], dim=0)
    return low, high


def stats_percentiles(x: torch.Tensor) -> dict[str, float]:
    x = x.reshape(-1)
    return {
        "p5": float(torch.quantile(x, 0.05).item()),
        "p50": float(torch.quantile(x, 0.50).item()),
        "p95": float(torch.quantile(x, 0.95).item()),
    }


def tensor_distribution_summary(
    x: torch.Tensor,
    *,
    reference: ReferenceStats | None = None,
    max_cov_dim_for_json: int = 64,
) -> dict[str, Any]:
    x = x.reshape(-1, x.shape[-1])
    norms = x.norm(dim=-1)
    per_dim_mean = x.mean(dim=0)
    per_dim_std = x.std(dim=0, unbiased=False)
    out: dict[str, Any] = {
        "num_samples": int(x.size(0)),
        "mean_norm": float(norms.mean().item()),
        "std_norm": float(norms.std(unbiased=False).item()),
        "per_dim_mean": per_dim_mean.detach().cpu().tolist(),
        "per_dim_std": per_dim_std.detach().cpu().tolist(),
        "diag_variance": (per_dim_std.square()).detach().cpu().tolist(),
    }
    if x.size(1) <= max_cov_dim_for_json:
        centered = x - per_dim_mean
        cov = (centered.T @ centered) / max(1, int(x.size(0)) - 1)
        out["covariance"] = cov.detach().cpu().tolist()
    if reference is not None:
        md2 = mahalanobis_sq(x, reference.mean, reference.inv_cov)
        out["mahalanobis"] = stats_percentiles(md2)
        out["mahalanobis_mean"] = float(md2.mean().item())
    return out


def build_chunk_sequences(
    ctx: DiagnosticContext,
    starts: np.ndarray,
    spans_tokens: Sequence[int],
) -> dict[str, Any]:
    spans_tokens = [int(x) for x in spans_tokens]
    raw_spans = [span * ctx.group for span in spans_tokens]
    total_raw = int(sum(raw_spans))
    starts = np.asarray(starts, dtype=np.int64)
    b = len(starts)
    h = len(spans_tokens)

    z_init = encode_row_latents(ctx, starts)

    macro_steps: list[torch.Tensor] = []
    target_steps: list[torch.Tensor] = []
    step_start_idx = starts.copy()
    step_target_idx: list[np.ndarray] = []
    cursor = np.zeros_like(starts)
    for raw_span, span_tokens in zip(raw_spans, spans_tokens):
        chunks_raw = np.stack(
            [ctx.action[s + cur : s + cur + raw_span] for s, cur in zip(starts, cursor, strict=True)],
            axis=0,
        )
        chunks_tok = preprocess_actions(
            chunks_raw,
            ctx.action_scaler,
            chunk_len_tokens=span_tokens,
            group=ctx.group,
        )
        macro_steps.append(encode_macro_actions(ctx.model, chunks_tok, device=ctx.device))
        cursor = cursor + raw_span
        target_idx = starts + cursor
        step_target_idx.append(target_idx.copy())
        target_steps.append(encode_row_latents(ctx, target_idx))

    macro_seq = torch.stack(macro_steps, dim=1) if h > 1 else macro_steps[0].unsqueeze(1)
    target_seq = torch.stack(target_steps, dim=1) if h > 1 else target_steps[0].unsqueeze(1)
    return {
        "starts": starts,
        "spans_tokens": spans_tokens,
        "raw_spans": raw_spans,
        "total_raw": total_raw,
        "z_init": z_init,
        "macro_seq": macro_seq,
        "target_seq": target_seq,
        "step_target_idx": step_target_idx,
        "step_start_idx": [starts + sum(raw_spans[:i]) for i in range(h)],
    }


def high_level_teacher_forced_rollout(
    model: torch.nn.Module,
    z_init: torch.Tensor,
    macro_seq: torch.Tensor,
    true_targets: torch.Tensor,
) -> torch.Tensor:
    h = int(macro_seq.size(1))
    z_prev = z_init
    preds = []
    for step in range(h):
        pred = rollout_one_high(model, z_prev, macro_seq[:, step])
        preds.append(pred)
        z_prev = true_targets[:, step]
    return torch.stack(preds, dim=1)


@torch.inference_mode()
def high_level_cem_optimize(
    *,
    model: torch.nn.Module,
    z_init: torch.Tensor,
    z_goal: torch.Tensor,
    refs: Sequence[ReferenceStats],
    num_samples: int,
    n_steps: int,
    topk: int,
    elite_frac: float,
    bound_mode: str,
) -> dict[str, Any]:
    b = int(z_init.size(0))
    h = len(refs)
    d_l = int(refs[0].mean.numel())
    mu = torch.stack([ref.mean for ref in refs], dim=0).unsqueeze(0).expand(b, -1, -1).clone()
    sigma = torch.stack([ref.std for ref in refs], dim=0).unsqueeze(0).expand(b, -1, -1).clone()
    elite_k = int(topk) if int(topk) > 0 else max(1, int(num_samples * elite_frac))
    low, high = expand_bounds(refs, bound_mode)
    if low is not None and high is not None:
        low = low.unsqueeze(0)
        high = high.unsqueeze(0)

    final_candidates = None
    final_cost = None
    final_elites = None
    final_elite_cost = None

    for _ in range(int(n_steps)):
        eps = torch.randn((b, num_samples, h, d_l), device=z_init.device, dtype=z_init.dtype)
        cand = mu[:, None, :, :] + sigma[:, None, :, :] * eps
        if low is not None and high is not None:
            cand = torch.max(torch.min(cand, high[:, None, :, :]), low[:, None, :, :])

        pred = model.rollout_high(z_init, cand)
        z_final = pred[:, :, -1, :]
        cost = (z_final - z_goal[:, None, :]).pow(2).mean(dim=-1)
        elite_idx = torch.topk(cost, k=min(elite_k, num_samples), dim=1, largest=False).indices
        expand_idx = elite_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, d_l)
        elite = torch.gather(cand, 1, expand_idx)
        elite_cost = torch.gather(cost, 1, elite_idx)

        mu = elite.mean(dim=1)
        sigma = elite.std(dim=1, unbiased=False).clamp_min(1e-4)
        final_candidates = cand
        final_cost = cost
        final_elites = elite
        final_elite_cost = elite_cost

    assert final_candidates is not None
    assert final_cost is not None
    assert final_elites is not None
    assert final_elite_cost is not None
    best_idx = torch.argmin(final_elite_cost, dim=1)
    batch_idx = torch.arange(b, device=z_init.device)
    best_seq = final_elites[batch_idx, best_idx]
    mean_seq = mu
    return {
        "selected_mean": mean_seq,
        "selected_best": best_seq,
        "final_candidates": final_candidates,
        "final_cost": final_cost,
        "final_elites": final_elites,
        "final_elite_cost": final_elite_cost,
    }


@torch.inference_mode()
def low_level_cem_optimize(
    *,
    model: torch.nn.Module,
    z_init: torch.Tensor,
    z_subgoal: torch.Tensor,
    action_ref: dict[str, torch.Tensor],
    horizon: int,
    num_samples: int,
    n_steps: int,
    topk: int,
    elite_frac: float,
    bound_mode: str,
) -> dict[str, Any]:
    b = int(z_init.size(0))
    a_dim = int(action_ref["mean"].numel())
    mu = action_ref["mean"].view(1, 1, a_dim).expand(b, horizon, a_dim).clone()
    sigma = action_ref["std"].view(1, 1, a_dim).expand(b, horizon, a_dim).clone()
    elite_k = int(topk) if int(topk) > 0 else max(1, int(num_samples * elite_frac))
    low = high = None
    if bound_mode != "none":
        low = action_ref["q05"].view(1, 1, a_dim)
        high = action_ref["q95"].view(1, 1, a_dim)

    final_candidates = None
    final_cost = None
    final_final_latent = None

    for _ in range(int(n_steps)):
        eps = torch.randn((b, num_samples, horizon, a_dim), device=z_init.device, dtype=z_init.dtype)
        cand = mu[:, None, :, :] + sigma[:, None, :, :] * eps
        if low is not None and high is not None:
            cand = torch.max(torch.min(cand, high[:, None, :, :]), low[:, None, :, :])

        a_hist = torch.zeros((b, num_samples, 1, a_dim), device=z_init.device, dtype=z_init.dtype)
        pred = model.rollout_low(z_init, a_hist, cand)
        z_final = pred[:, :, -1, :]
        cost = (z_final - z_subgoal[:, None, :]).pow(2).mean(dim=-1)
        elite_idx = torch.topk(cost, k=min(elite_k, num_samples), dim=1, largest=False).indices
        expand_idx = elite_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, horizon, a_dim)
        elite = torch.gather(cand, 1, expand_idx)
        mu = elite.mean(dim=1)
        sigma = elite.std(dim=1, unbiased=False).clamp_min(1e-4)
        final_candidates = cand
        final_cost = cost
        final_final_latent = z_final

    assert final_candidates is not None
    assert final_cost is not None
    assert final_final_latent is not None
    best_idx = torch.argmin(final_cost, dim=1)
    batch_idx = torch.arange(b, device=z_init.device)
    best_seq = final_candidates[batch_idx, best_idx]
    best_final_latent = final_final_latent[batch_idx, best_idx]
    return {
        "selected_mean": mu,
        "selected_best": best_seq,
        "final_candidates": final_candidates,
        "final_cost": final_cost,
        "best_final_latent": best_final_latent,
    }


def same_trajectory_future_latents(
    ctx: DiagnosticContext,
    stage_starts: np.ndarray,
    future_raw_span: int,
) -> tuple[list[np.ndarray], list[torch.Tensor]]:
    index_lists: list[np.ndarray] = []
    tensor_lists: list[torch.Tensor] = []
    for start in np.asarray(stage_starts, dtype=np.int64):
        idx = np.arange(start + 1, start + future_raw_span + 1, dtype=np.int64)
        index_lists.append(idx)
        tensor_lists.append(encode_row_latents(ctx, idx))
    return index_lists, tensor_lists


def min_distance_to_pool(query: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(query, pool)
    return d.min(dim=1).values


def per_example_same_traj_min_distance(query: torch.Tensor, pools: Sequence[torch.Tensor]) -> torch.Tensor:
    out = []
    for q, pool in zip(query, pools, strict=True):
        out.append(torch.cdist(q.unsqueeze(0), pool).min())
    return torch.stack(out, dim=0)


def maybe_save_npz(path: str | None, arrays: dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_serializable(payload), indent=2))


def append_tsv(path: str | None, columns: Sequence[str], row: dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    values = {}
    for key in columns:
        value = row.get(key, "")
        if isinstance(value, (dict, list, tuple)):
            values[key] = json.dumps(to_serializable(value), separators=(",", ":"))
        else:
            values[key] = value
    import fcntl

    with out_path.open("a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        has_content = bool(f.read(1))
        f.seek(0, 2)
        writer = csv.DictWriter(f, fieldnames=list(columns), delimiter="\t")
        if not has_content:
            writer.writeheader()
        writer.writerow(values)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def finalize_result(
    ctx: DiagnosticContext,
    result: dict[str, Any],
    *,
    tsv_columns: Sequence[str],
    tsv_row: dict[str, Any],
    npz_arrays: dict[str, Any] | None = None,
) -> dict[str, Any]:
    write_json(ctx.cfg.save_json, result)
    append_tsv(ctx.cfg.append_tsv, tsv_columns, tsv_row)
    if npz_arrays:
        maybe_save_npz(ctx.cfg.save_npz, npz_arrays)
    return result


def base_result(ctx: DiagnosticContext) -> dict[str, Any]:
    run_name, checkpoint_epoch = parse_policy_run_info(ctx.cfg.policy)
    return {
        "run_name": run_name,
        "checkpoint_epoch": checkpoint_epoch,
        "policy": ctx.cfg.policy,
        "dataset_name": ctx.cfg.dataset_name,
        "experiment_kind": ctx.cfg.experiment_kind,
        "device": str(ctx.device),
        "seed": int(ctx.cfg.seed),
        "goal_offset_steps": int(ctx.cfg.goal_offset_steps),
        "goal_tokens": int(goal_tokens(ctx.cfg)),
        "high_horizon": int(ctx.cfg.high_horizon),
        "low_horizon": int(ctx.cfg.low_horizon),
        "latent_action_dim": int(ctx.latent_dim),
        "macro_input_dim": int(ctx.macro_input_dim),
        "low_action_input_dim": int(ctx.low_action_input_dim),
        "raw_action_dim": int(ctx.raw_action_dim),
        "group_factor": int(ctx.group),
        "result_status": "ok",
    }


def run_macro_action_manifold_diagnostic(cfg: DiagnosticConfig) -> dict[str, Any]:
    ctx = build_context(cfg)
    spans = partition_total(goal_tokens(cfg), int(cfg.high_horizon))
    refs = [build_macro_reference(ctx, span) for span in spans]
    total_raw = sum(span * ctx.group for span in spans)
    starts = sample_valid_starts_for_raw_span(ctx, raw_span=total_raw, n=cfg.num_eval_samples)
    seq = build_chunk_sequences(ctx, starts, spans)
    z_goal = seq["target_seq"][:, -1, :]

    cem = high_level_cem_optimize(
        model=ctx.model,
        z_init=seq["z_init"],
        z_goal=z_goal,
        refs=refs,
        num_samples=int(cfg.high_num_samples),
        n_steps=int(cfg.high_iters),
        topk=int(cfg.high_topk),
        elite_frac=float(cfg.cem_elite_frac),
        bound_mode=cfg.cem_bound_mode,
    )

    step_metrics = []
    for step, ref in enumerate(refs):
        selected_mean_step = cem["selected_mean"][:, step, :]
        selected_best_step = cem["selected_best"][:, step, :]
        elite_cloud_step = cem["final_elites"][:, :, step, :].reshape(-1, ctx.latent_dim)
        candidate_cloud_step = cem["final_candidates"][:, :, step, :].reshape(-1, ctx.latent_dim)
        true_step = seq["macro_seq"][:, step, :]
        step_metrics.append(
            {
                "step": step + 1,
                "span_tokens": int(ref.span_tokens),
                "dataset": tensor_distribution_summary(
                    ref.samples,
                    reference=ref,
                    max_cov_dim_for_json=cfg.max_cov_dim_for_json,
                ),
                "true_macro": tensor_distribution_summary(
                    true_step,
                    reference=ref,
                    max_cov_dim_for_json=cfg.max_cov_dim_for_json,
                ),
                "selected_mean": tensor_distribution_summary(
                    selected_mean_step,
                    reference=ref,
                    max_cov_dim_for_json=cfg.max_cov_dim_for_json,
                ),
                "selected_best": tensor_distribution_summary(
                    selected_best_step,
                    reference=ref,
                    max_cov_dim_for_json=cfg.max_cov_dim_for_json,
                ),
                "elite_cloud": tensor_distribution_summary(
                    elite_cloud_step,
                    reference=ref,
                    max_cov_dim_for_json=cfg.max_cov_dim_for_json,
                ),
                "candidate_cloud": tensor_distribution_summary(
                    candidate_cloud_step,
                    reference=ref,
                    max_cov_dim_for_json=cfg.max_cov_dim_for_json,
                ),
            }
        )

    result = base_result(ctx)
    result.update(
        {
            "spans_tokens": spans,
            "num_eval_samples": int(cfg.num_eval_samples),
            "num_empirical_chunks": int(cfg.num_empirical_chunks),
            "step_metrics": step_metrics,
        }
    )

    tsv_columns = [
        "experiment_kind",
        "policy",
        "goal_offset_steps",
        "high_horizon",
        "step1_span_tokens",
        "step1_dataset_mean_norm",
        "step1_selected_mean_norm",
        "step1_selected_best_mean_norm",
        "step1_elite_mean_norm",
        "step1_selected_mean_md2_p50",
        "step1_elite_md2_p50",
        "step2_span_tokens",
        "step2_dataset_mean_norm",
        "step2_selected_mean_norm",
        "step2_selected_best_mean_norm",
        "step2_elite_mean_norm",
        "step2_selected_mean_md2_p50",
        "step2_elite_md2_p50",
        "result_status",
    ]
    tsv_row: dict[str, Any] = {
        "experiment_kind": cfg.experiment_kind,
        "policy": cfg.policy,
        "goal_offset_steps": cfg.goal_offset_steps,
        "high_horizon": cfg.high_horizon,
        "result_status": "ok",
    }
    for step_idx, step_data in enumerate(step_metrics[:2], start=1):
        tsv_row[f"step{step_idx}_span_tokens"] = step_data["span_tokens"]
        tsv_row[f"step{step_idx}_dataset_mean_norm"] = step_data["dataset"]["mean_norm"]
        tsv_row[f"step{step_idx}_selected_mean_norm"] = step_data["selected_mean"]["mean_norm"]
        tsv_row[f"step{step_idx}_selected_best_mean_norm"] = step_data["selected_best"]["mean_norm"]
        tsv_row[f"step{step_idx}_elite_mean_norm"] = step_data["elite_cloud"]["mean_norm"]
        tsv_row[f"step{step_idx}_selected_mean_md2_p50"] = step_data["selected_mean"]["mahalanobis"]["p50"]
        tsv_row[f"step{step_idx}_elite_md2_p50"] = step_data["elite_cloud"]["mahalanobis"]["p50"]

    npz_arrays = {
        "starts": np.asarray(starts),
        "z_init": seq["z_init"].detach().cpu().numpy(),
        "target_seq": seq["target_seq"].detach().cpu().numpy(),
        "macro_seq": seq["macro_seq"].detach().cpu().numpy(),
        "selected_mean": cem["selected_mean"].detach().cpu().numpy(),
        "selected_best": cem["selected_best"].detach().cpu().numpy(),
        "final_elite_cost": cem["final_elite_cost"].detach().cpu().numpy(),
    }
    return finalize_result(ctx, result, tsv_columns=tsv_columns, tsv_row=tsv_row, npz_arrays=npz_arrays)


def run_high_level_teacher_vs_open_loop_diagnostic(cfg: DiagnosticConfig) -> dict[str, Any]:
    ctx = build_context(cfg)
    spans = partition_total(goal_tokens(cfg), int(cfg.high_horizon))
    refs = [build_macro_reference(ctx, span) for span in spans]
    total_raw = sum(span * ctx.group for span in spans)
    starts = sample_valid_starts_for_raw_span(ctx, raw_span=total_raw, n=cfg.num_eval_samples)
    seq = build_chunk_sequences(ctx, starts, spans)
    z_goal = seq["target_seq"][:, -1, :]

    teacher_pred = high_level_teacher_forced_rollout(ctx.model, seq["z_init"], seq["macro_seq"], seq["target_seq"])
    open_true_pred = ctx.model.rollout_high(seq["z_init"], seq["macro_seq"])[:, 0, :, :]
    cem = high_level_cem_optimize(
        model=ctx.model,
        z_init=seq["z_init"],
        z_goal=z_goal,
        refs=refs,
        num_samples=int(cfg.high_num_samples),
        n_steps=int(cfg.high_iters),
        topk=int(cfg.high_topk),
        elite_frac=float(cfg.cem_elite_frac),
        bound_mode=cfg.cem_bound_mode,
    )
    open_cem_pred = ctx.model.rollout_high(seq["z_init"], cem["selected_mean"])[:, 0, :, :]

    teacher_err = (teacher_pred - seq["target_seq"]).pow(2).mean(dim=-1)
    open_true_err = (open_true_pred - seq["target_seq"]).pow(2).mean(dim=-1)
    open_cem_err = (open_cem_pred - seq["target_seq"]).pow(2).mean(dim=-1)

    open_true_over_teacher = float(
        (open_true_err.mean() / teacher_err.mean().clamp_min(1e-8)).item()
    )
    open_cem_over_true = float(
        (open_cem_err.mean() / open_true_err.mean().clamp_min(1e-8)).item()
    )

    result = base_result(ctx)
    result.update(
        {
            "spans_tokens": spans,
            "teacher_forced_mse_per_step": teacher_err.mean(dim=0).detach().cpu().tolist(),
            "open_loop_true_mse_per_step": open_true_err.mean(dim=0).detach().cpu().tolist(),
            "open_loop_cem_mse_per_step": open_cem_err.mean(dim=0).detach().cpu().tolist(),
            "teacher_forced_mse_mean": float(teacher_err.mean().item()),
            "open_loop_true_mse_mean": float(open_true_err.mean().item()),
            "open_loop_cem_mse_mean": float(open_cem_err.mean().item()),
            "open_loop_true_over_teacher": open_true_over_teacher,
            "open_loop_cem_over_open_true": open_cem_over_true,
            "planner_prior_issue_flag": bool(open_cem_over_true > 1.5),
            "high_predictor_instability_flag": bool(open_true_over_teacher > 1.5),
        }
    )

    tsv_columns = [
        "experiment_kind",
        "policy",
        "goal_offset_steps",
        "high_horizon",
        "teacher_forced_mse_mean",
        "open_loop_true_mse_mean",
        "open_loop_cem_mse_mean",
        "open_loop_true_over_teacher",
        "open_loop_cem_over_open_true",
        "planner_prior_issue_flag",
        "high_predictor_instability_flag",
        "step1_teacher_mse",
        "step1_open_true_mse",
        "step1_open_cem_mse",
        "step2_teacher_mse",
        "step2_open_true_mse",
        "step2_open_cem_mse",
        "result_status",
    ]
    tsv_row: dict[str, Any] = {
        "experiment_kind": cfg.experiment_kind,
        "policy": cfg.policy,
        "goal_offset_steps": cfg.goal_offset_steps,
        "high_horizon": cfg.high_horizon,
        "teacher_forced_mse_mean": result["teacher_forced_mse_mean"],
        "open_loop_true_mse_mean": result["open_loop_true_mse_mean"],
        "open_loop_cem_mse_mean": result["open_loop_cem_mse_mean"],
        "open_loop_true_over_teacher": open_true_over_teacher,
        "open_loop_cem_over_open_true": open_cem_over_true,
        "planner_prior_issue_flag": result["planner_prior_issue_flag"],
        "high_predictor_instability_flag": result["high_predictor_instability_flag"],
        "result_status": "ok",
    }
    for step in range(min(2, len(spans))):
        tsv_row[f"step{step + 1}_teacher_mse"] = result["teacher_forced_mse_per_step"][step]
        tsv_row[f"step{step + 1}_open_true_mse"] = result["open_loop_true_mse_per_step"][step]
        tsv_row[f"step{step + 1}_open_cem_mse"] = result["open_loop_cem_mse_per_step"][step]

    npz_arrays = {
        "starts": np.asarray(starts),
        "z_init": seq["z_init"].detach().cpu().numpy(),
        "target_seq": seq["target_seq"].detach().cpu().numpy(),
        "teacher_pred": teacher_pred.detach().cpu().numpy(),
        "open_true_pred": open_true_pred.detach().cpu().numpy(),
        "open_cem_pred": open_cem_pred.detach().cpu().numpy(),
        "teacher_err": teacher_err.detach().cpu().numpy(),
        "open_true_err": open_true_err.detach().cpu().numpy(),
        "open_cem_err": open_cem_err.detach().cpu().numpy(),
        "selected_mean": cem["selected_mean"].detach().cpu().numpy(),
    }
    return finalize_result(ctx, result, tsv_columns=tsv_columns, tsv_row=tsv_row, npz_arrays=npz_arrays)


def run_dataset_subgoal_reachability_diagnostic(cfg: DiagnosticConfig) -> dict[str, Any]:
    ctx = build_context(cfg)
    offsets = tuple(int(x) for x in cfg.subgoal_offsets)
    max_raw = max(offsets) * ctx.group
    starts = sample_valid_starts_for_raw_span(ctx, raw_span=max_raw, n=cfg.num_eval_samples)
    z_init = encode_row_latents(ctx, starts)
    action_ref = build_action_token_reference(ctx)

    offset_results = []
    for offset in offsets:
        target_idx = starts + offset * ctx.group
        z_target = encode_row_latents(ctx, target_idx)
        low = low_level_cem_optimize(
            model=ctx.model,
            z_init=z_init,
            z_subgoal=z_target,
            action_ref=action_ref,
            horizon=int(cfg.low_horizon),
            num_samples=int(cfg.low_num_samples),
            n_steps=int(cfg.low_iters),
            topk=int(cfg.low_topk),
            elite_frac=float(cfg.cem_elite_frac),
            bound_mode=cfg.cem_bound_mode,
        )
        final_err = (low["best_final_latent"] - z_target).pow(2).mean(dim=-1)
        offset_results.append(
            {
                "offset_tokens": int(offset),
                "terminal_latent_error_mean": float(final_err.mean().item()),
                "best_cem_terminal_cost_mean": float(low["final_cost"].min(dim=1).values.mean().item()),
                "selected_mean_action_norm": float(low["selected_mean"].norm(dim=-1).mean().item()),
                "selected_best_action_norm": float(low["selected_best"].norm(dim=-1).mean().item()),
            }
        )

    result = base_result(ctx)
    result.update(
        {
            "offset_results": offset_results,
            "offsets_tokens": list(offsets),
            "terminal_latent_error_overall_mean": float(
                np.mean([item["terminal_latent_error_mean"] for item in offset_results])
            ),
        }
    )

    tsv_columns = [
        "experiment_kind",
        "policy",
        "goal_offset_steps",
        "low_horizon",
        "offset2_terminal_error_mean",
        "offset3_terminal_error_mean",
        "offset5_terminal_error_mean",
        "overall_terminal_error_mean",
        "result_status",
    ]
    tsv_row: dict[str, Any] = {
        "experiment_kind": cfg.experiment_kind,
        "policy": cfg.policy,
        "goal_offset_steps": cfg.goal_offset_steps,
        "low_horizon": cfg.low_horizon,
        "overall_terminal_error_mean": result["terminal_latent_error_overall_mean"],
        "result_status": "ok",
    }
    for item in offset_results:
        tsv_row[f"offset{item['offset_tokens']}_terminal_error_mean"] = item["terminal_latent_error_mean"]

    npz_arrays = {
        "starts": np.asarray(starts),
        "z_init": z_init.detach().cpu().numpy(),
    }
    return finalize_result(ctx, result, tsv_columns=tsv_columns, tsv_row=tsv_row, npz_arrays=npz_arrays)


def run_generated_subgoal_reachability_diagnostic(cfg: DiagnosticConfig) -> dict[str, Any]:
    ctx = build_context(cfg)
    spans = partition_total(goal_tokens(cfg), int(cfg.high_horizon))
    refs = [build_macro_reference(ctx, span) for span in spans]
    total_raw = sum(span * ctx.group for span in spans)
    starts = sample_valid_starts_for_raw_span(ctx, raw_span=total_raw, n=cfg.num_eval_samples)
    seq = build_chunk_sequences(ctx, starts, spans)
    z_goal = seq["target_seq"][:, -1, :]
    cem = high_level_cem_optimize(
        model=ctx.model,
        z_init=seq["z_init"],
        z_goal=z_goal,
        refs=refs,
        num_samples=int(cfg.high_num_samples),
        n_steps=int(cfg.high_iters),
        topk=int(cfg.high_topk),
        elite_frac=float(cfg.cem_elite_frac),
        bound_mode=cfg.cem_bound_mode,
    )
    pred_seq = ctx.model.rollout_high(seq["z_init"], cem["selected_mean"])[:, 0, :, :]
    action_ref = build_action_token_reference(ctx)
    _, dataset_pool = get_reference_latent_pool(ctx, int(cfg.reference_latent_pool_size))
    state_ref = build_reference_stats(dataset_pool, span_tokens=int(goal_tokens(cfg)))

    step_results = []
    cumulative_raw = 0
    for step, span in enumerate(spans):
        cumulative_raw += span * ctx.group
        stage_start_idx = starts + sum(spans[:step]) * ctx.group
        stage_init = seq["z_init"] if step == 0 else seq["target_seq"][:, step - 1, :]
        generated_subgoal = pred_seq[:, step, :]
        low = low_level_cem_optimize(
            model=ctx.model,
            z_init=stage_init,
            z_subgoal=generated_subgoal,
            action_ref=action_ref,
            horizon=int(cfg.low_horizon),
            num_samples=int(cfg.low_num_samples),
            n_steps=int(cfg.low_iters),
            topk=int(cfg.low_topk),
            elite_frac=float(cfg.cem_elite_frac),
            bound_mode=cfg.cem_bound_mode,
        )

        same_idx, same_latents = same_trajectory_future_latents(
            ctx,
            stage_starts=stage_start_idx,
            future_raw_span=max(span * ctx.group, 1),
        )
        dataset_dist = min_distance_to_pool(generated_subgoal, dataset_pool)
        same_traj_dist = per_example_same_traj_min_distance(generated_subgoal, same_latents)
        md2 = mahalanobis_sq(generated_subgoal, state_ref.mean, state_ref.inv_cov)
        terminal_err = (low["best_final_latent"] - generated_subgoal).pow(2).mean(dim=-1)

        step_results.append(
            {
                "step": step + 1,
                "span_tokens": int(span),
                "low_level_best_cem_terminal_cost_mean": float(low["final_cost"].min(dim=1).values.mean().item()),
                "achieved_terminal_latent_error_mean": float(terminal_err.mean().item()),
                "nearest_dataset_latent_distance_mean": float(dataset_dist.mean().item()),
                "nearest_same_trajectory_future_distance_mean": float(same_traj_dist.mean().item()),
                "generated_subgoal_mean_norm": float(generated_subgoal.norm(dim=-1).mean().item()),
                "generated_subgoal_mahalanobis": stats_percentiles(md2),
            }
        )

    result = base_result(ctx)
    result.update(
        {
            "spans_tokens": spans,
            "step_results": step_results,
        }
    )

    tsv_columns = [
        "experiment_kind",
        "policy",
        "goal_offset_steps",
        "high_horizon",
        "low_horizon",
        "step1_terminal_cost_mean",
        "step1_terminal_error_mean",
        "step1_dataset_distance_mean",
        "step1_same_traj_distance_mean",
        "step2_terminal_cost_mean",
        "step2_terminal_error_mean",
        "step2_dataset_distance_mean",
        "step2_same_traj_distance_mean",
        "result_status",
    ]
    tsv_row: dict[str, Any] = {
        "experiment_kind": cfg.experiment_kind,
        "policy": cfg.policy,
        "goal_offset_steps": cfg.goal_offset_steps,
        "high_horizon": cfg.high_horizon,
        "low_horizon": cfg.low_horizon,
        "result_status": "ok",
    }
    for step_data in step_results[:2]:
        step_idx = int(step_data["step"])
        tsv_row[f"step{step_idx}_terminal_cost_mean"] = step_data["low_level_best_cem_terminal_cost_mean"]
        tsv_row[f"step{step_idx}_terminal_error_mean"] = step_data["achieved_terminal_latent_error_mean"]
        tsv_row[f"step{step_idx}_dataset_distance_mean"] = step_data["nearest_dataset_latent_distance_mean"]
        tsv_row[f"step{step_idx}_same_traj_distance_mean"] = step_data["nearest_same_trajectory_future_distance_mean"]

    npz_arrays = {
        "starts": np.asarray(starts),
        "z_init": seq["z_init"].detach().cpu().numpy(),
        "target_seq": seq["target_seq"].detach().cpu().numpy(),
        "selected_mean_macro_seq": cem["selected_mean"].detach().cpu().numpy(),
        "generated_subgoal_seq": pred_seq.detach().cpu().numpy(),
    }
    return finalize_result(ctx, result, tsv_columns=tsv_columns, tsv_row=tsv_row, npz_arrays=npz_arrays)


def run_diagnostic(cfg: DiagnosticConfig) -> dict[str, Any]:
    if cfg.experiment_kind == "macro_manifold":
        return run_macro_action_manifold_diagnostic(cfg)
    if cfg.experiment_kind == "teacher_vs_open_loop":
        return run_high_level_teacher_vs_open_loop_diagnostic(cfg)
    if cfg.experiment_kind == "dataset_subgoal_reachability":
        return run_dataset_subgoal_reachability_diagnostic(cfg)
    if cfg.experiment_kind == "generated_subgoal_reachability":
        return run_generated_subgoal_reachability_diagnostic(cfg)
    raise ValueError(f"Unsupported experiment_kind={cfg.experiment_kind}")
