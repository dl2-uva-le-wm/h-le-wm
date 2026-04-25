#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import baseline_adapter as _baseline_adapter

# Backward-compatibility for torch.load on object checkpoints saved by hi_train:
# those pickles may reference classes under the dynamic module name
# `_baseline_lewm_module` (created by baseline_adapter). Touch one exported
# symbol so baseline_adapter registers that dynamic module in sys.modules
# before AutoCostModel unpickles.
_ = _baseline_adapter.ARPredictor


def img_transform(img_size: int):
    # Prefer stable_pretraining ImageNet stats (used elsewhere in this repo).
    # Keep a hard fallback for environments where that symbol layout changes.
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
    return pool[pick]


def get_row_data_safe(dataset, row_idx: np.ndarray) -> dict:
    """Fetch rows with h5py-compatible sorted indexing, then restore original order."""
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
    # actions_raw: (B, raw_chunk_len, raw_dim)
    b, raw_chunk_len, raw_dim = actions_raw.shape
    flat = actions_raw.reshape(-1, raw_dim)
    norm = action_scaler.transform(flat).reshape(b, raw_chunk_len, raw_dim)
    if group == 1:
        return norm
    return norm.reshape(b, chunk_len_tokens, raw_dim * group)


def encode_pixels_last(model: torch.nn.Module, pixels_bhwc: np.ndarray, tfm, device: torch.device):
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
    # z_init: (B, D), macro_actions: (B, D_l)
    pred = model.rollout_high(z_init, macro_actions.unsqueeze(1))
    return pred[:, 0, 0, :]


def mahalanobis_sq(x: torch.Tensor, mean: torch.Tensor, inv_cov: torch.Tensor) -> torch.Tensor:
    diff = x - mean
    return torch.einsum("bd,dd,bd->b", diff, inv_cov, diff)


@torch.inference_mode()
def cem_optimize_macro_actions(
    *,
    model: torch.nn.Module,
    z_init: torch.Tensor,
    z_goal: torch.Tensor,
    mu0: torch.Tensor,
    sigma0: torch.Tensor,
    low: torch.Tensor | None,
    high: torch.Tensor | None,
    num_samples: int,
    n_steps: int,
    elite_frac: float,
) -> torch.Tensor:
    # z_init / z_goal: (B, D), output: (B, D_l)
    b = z_init.size(0)
    d_l = mu0.numel()
    mu = mu0.unsqueeze(0).expand(b, -1).clone()
    sigma = sigma0.unsqueeze(0).expand(b, -1).clone()
    k = max(1, int(num_samples * elite_frac))

    for _ in range(n_steps):
        eps = torch.randn((b, num_samples, d_l), device=z_init.device, dtype=z_init.dtype)
        cand = mu[:, None, :] + sigma[:, None, :] * eps
        if low is not None and high is not None:
            cand = torch.max(torch.min(cand, high.view(1, 1, -1)), low.view(1, 1, -1))

        pred = model.rollout_high(z_init, cand.unsqueeze(2))[:, :, 0, :]
        cost = (pred - z_goal[:, None, :]).pow(2).mean(dim=-1)
        elite_idx = torch.topk(cost, k=k, dim=1, largest=False).indices
        elite = torch.gather(cand, 1, elite_idx.unsqueeze(-1).expand(-1, -1, d_l))

        mu = elite.mean(dim=1)
        sigma = elite.std(dim=1, unbiased=False).clamp_min(1e-4)

    return mu


def build_action_scaler(dataset) -> preprocessing.StandardScaler:
    action = np.asarray(dataset.get_col_data("action"))
    valid = ~np.isnan(action).any(axis=1)
    scaler = preprocessing.StandardScaler()
    scaler.fit(action[valid])
    return scaler


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Diagnostic for macro-action manifold mismatch:\n"
            "1) true dataset macro-actions -> one-step high-level prediction error\n"
            "2) CEM-optimized macro-actions -> one-step prediction + off-manifold metrics"
        )
    )
    p.add_argument("--policy", required=True, help="Policy path for AutoCostModel (relative to STABLEWM_HOME).")
    p.add_argument("--dataset-name", default="pusht_expert_train")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument(
        "--chunk-len-tokens",
        type=int,
        default=5,
        help="Macro chunk length in grouped-action tokens (not raw env steps).",
    )
    p.add_argument("--num-eval-samples", type=int, default=256)
    p.add_argument("--num-empirical-chunks", type=int, default=4096)
    p.add_argument("--cem-samples", type=int, default=900)
    p.add_argument("--cem-iters", type=int, default=20)
    p.add_argument("--cem-elite-frac", type=float, default=0.1)
    p.add_argument(
        "--cem-bound-mode",
        default="none",
        choices=["none", "q01_q99", "q05_q95"],
        help="Optional clamp for CEM candidate macro-actions.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--save-json", default=None, help="Optional output JSON path.")
    args = p.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    rng = np.random.default_rng(args.seed)

    dataset = swm.data.HDF5Dataset(
        args.dataset_name,
        keys_to_cache=["action"],
        cache_dir=Path(args.cache_dir) if args.cache_dir else Path(swm.data.utils.get_cache_dir()),
    )
    episode_col = get_episode_col_name(dataset)
    episode_ids = np.asarray(dataset.get_col_data(episode_col))
    step_idx = np.asarray(dataset.get_col_data("step_idx")) if "step_idx" in dataset.column_names else None
    action = np.asarray(dataset.get_col_data("action"))
    seq_len = int(action.shape[0])
    raw_action_dim = int(action.shape[1])

    model = swm.policy.AutoCostModel(args.policy).to(device).eval()
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
    raw_chunk_len = int(args.chunk_len_tokens) * group

    valid_starts = contiguous_valid_starts(
        episode_ids=episode_ids,
        step_idx=step_idx,
        seq_len=seq_len,
        chunk_len=raw_chunk_len + 1,  # need target frame at start + raw_chunk_len
    )
    if valid_starts.size == 0:
        raise ValueError("No valid contiguous starts found for the requested chunk length.")

    scaler = build_action_scaler(dataset)
    tfm = img_transform(args.img_size)

    # Build empirical macro-action latent distribution.
    emp_starts = sample_indices(rng, valid_starts, args.num_empirical_chunks)
    emp_chunks_raw = np.stack([action[s : s + raw_chunk_len] for s in emp_starts], axis=0)
    emp_chunks_tok = preprocess_actions(
        emp_chunks_raw,
        scaler,
        chunk_len_tokens=int(args.chunk_len_tokens),
        group=group,
    )
    macro_emp = encode_macro_actions(model, emp_chunks_tok, device=device)
    emp_mean = macro_emp.mean(dim=0)
    emp_std = macro_emp.std(dim=0, unbiased=False).clamp_min(1e-3)
    emp_q01 = torch.quantile(macro_emp, 0.01, dim=0)
    emp_q05 = torch.quantile(macro_emp, 0.05, dim=0)
    emp_q95 = torch.quantile(macro_emp, 0.95, dim=0)
    emp_q99 = torch.quantile(macro_emp, 0.99, dim=0)

    x_center = macro_emp - emp_mean
    cov = (x_center.T @ x_center) / max(1, (macro_emp.size(0) - 1))
    cov = cov + 1e-4 * torch.eye(latent_dim, device=device, dtype=cov.dtype)
    inv_cov = torch.linalg.pinv(cov)
    md2_emp = mahalanobis_sq(macro_emp, emp_mean, inv_cov)
    md2_emp_q95 = torch.quantile(md2_emp, 0.95)

    # Eval starts for comparison.
    eval_starts = sample_indices(rng, valid_starts, args.num_eval_samples)
    idx_start = eval_starts
    idx_goal = eval_starts + raw_chunk_len

    rows_start = get_row_data_safe(dataset, idx_start)
    rows_goal = get_row_data_safe(dataset, idx_goal)
    pix_start = np.asarray(rows_start["pixels"])
    pix_goal = np.asarray(rows_goal["pixels"])

    z_start = encode_pixels_last(model, pix_start, tfm, device)
    z_goal = encode_pixels_last(model, pix_goal, tfm, device)

    eval_chunks_raw = np.stack([action[s : s + raw_chunk_len] for s in eval_starts], axis=0)
    eval_chunks_tok = preprocess_actions(
        eval_chunks_raw,
        scaler,
        chunk_len_tokens=int(args.chunk_len_tokens),
        group=group,
    )

    macro_true = encode_macro_actions(model, eval_chunks_tok, device=device)
    pred_true = rollout_one_high(model, z_start, macro_true)
    err_true = (pred_true - z_goal).pow(2).mean(dim=-1)

    cem_low = None
    cem_high = None
    if args.cem_bound_mode == "q01_q99":
        cem_low, cem_high = emp_q01, emp_q99
    elif args.cem_bound_mode == "q05_q95":
        cem_low, cem_high = emp_q05, emp_q95

    macro_cem = cem_optimize_macro_actions(
        model=model,
        z_init=z_start,
        z_goal=z_goal,
        mu0=emp_mean,
        sigma0=emp_std,
        low=cem_low,
        high=cem_high,
        num_samples=int(args.cem_samples),
        n_steps=int(args.cem_iters),
        elite_frac=float(args.cem_elite_frac),
    )
    pred_cem = rollout_one_high(model, z_start, macro_cem)
    err_cem = (pred_cem - z_goal).pow(2).mean(dim=-1)

    in_box_frac = ((macro_cem >= emp_q05) & (macro_cem <= emp_q95)).float().mean(dim=-1)
    md2_true = mahalanobis_sq(macro_true, emp_mean, inv_cov)
    md2_cem = mahalanobis_sq(macro_cem, emp_mean, inv_cov)

    summary = {
        "policy": args.policy,
        "dataset_name": args.dataset_name,
        "device": str(device),
        "latent_action_dim": latent_dim,
        "macro_input_dim": macro_input_dim,
        "raw_action_dim": raw_action_dim,
        "group_factor": group,
        "chunk_len_tokens": int(args.chunk_len_tokens),
        "raw_chunk_len": raw_chunk_len,
        "num_eval_samples": int(args.num_eval_samples),
        "num_empirical_chunks": int(args.num_empirical_chunks),
        "cem_bound_mode": args.cem_bound_mode,
        "true_macro_pred_mse_mean": float(err_true.mean().item()),
        "cem_macro_pred_mse_mean": float(err_cem.mean().item()),
        "pred_mse_ratio_cem_over_true": float((err_cem.mean() / err_true.mean().clamp_min(1e-8)).item()),
        "cem_in_q05_q95_box_dim_frac_mean": float(in_box_frac.mean().item()),
        "md2_emp_q95": float(md2_emp_q95.item()),
        "md2_true_mean": float(md2_true.mean().item()),
        "md2_cem_mean": float(md2_cem.mean().item()),
        "cem_md2_above_emp_q95_frac": float((md2_cem > md2_emp_q95).float().mean().item()),
    }

    print("=== Macro-Action Manifold Diagnostic ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("")
    print("Interpretation rule-of-thumb:")
    print(
        "- If true_macro_pred_mse_mean is low but cem_macro_pred_mse_mean is much higher,\n"
        "  and cem_md2_above_emp_q95_frac is high, this supports off-manifold CEM macro-actions."
    )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved JSON summary to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
