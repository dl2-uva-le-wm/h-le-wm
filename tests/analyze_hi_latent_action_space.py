from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel as swm
import torch

# Ensure repo-root modules (e.g., baseline_adapter.py) are importable when this
# script is launched as `python tests/analyze_hi_latent_action_space.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import baseline_adapter as _baseline_adapter

# Backward compatibility for torch.load on object checkpoints that reference
# baseline classes under dynamic module names.
_ = _baseline_adapter.ARPredictor


@dataclass(frozen=True)
class AnalysisStats:
    checkpoint: str
    dataset: str
    cache_dir: str
    num_chunks: int
    latent_dim: int
    min_chunk_len: int
    max_chunk_len: int
    mean_chunk_len: float
    norm_mean: float
    norm_std: float
    norm_p05: float
    norm_p50: float
    norm_p95: float
    length_norm_corr: float
    pc1_explained_ratio: float
    pc2_explained_ratio: float
    effective_rank: float
    effective_rank_fraction: float
    isotropy_ratio: float
    cosine_mean: float
    cosine_std: float
    cosine_p05: float
    cosine_p95: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze high-level latent action space by encoding action chunks with "
            "the checkpoint's latent_action_encoder and generating plots/summary."
        )
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Explicit checkpoint path.")
    parser.add_argument(
        "--runs-root",
        type=str,
        default="",
        help="Directory containing run folders with object checkpoints.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Run name prefix used in checkpoint filename (<run-name>_epoch_N_object.ckpt).",
    )
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=10,
        help="Epoch index to select when checkpoint is not given explicitly.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="pusht_expert_train",
        help="StableWM dataset name used to source primitive actions.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="StableWM cache dir (defaults to STABLEWM_HOME or package default).",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=12000,
        help="Number of variable-length action chunks to encode.",
    )
    parser.add_argument(
        "--min-chunk-len",
        type=int,
        default=1,
        help="Minimum action chunk length.",
    )
    parser.add_argument(
        "--max-chunk-len",
        type=int,
        default=15,
        help="Maximum action chunk length before model max_seq_len clipping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size used while encoding chunks on CPU.",
    )
    parser.add_argument(
        "--num-cosine-pairs",
        type=int,
        default=50000,
        help="How many random embedding pairs to use for cosine-sim diagnostics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3072,
        help="RNG seed for chunk sampling and pair sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (defaults to outputs/latent_analysis/<checkpoint_stem>).",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "hi_train.py").exists():
            return candidate
    raise FileNotFoundError("Could not resolve repo root containing hi_train.py")


def resolve_cache_dir(arg_cache_dir: str) -> Path:
    if arg_cache_dir:
        return Path(arg_cache_dir).expanduser().resolve()
    stablewm_home = os.getenv("STABLEWM_HOME", "").strip()
    if stablewm_home:
        return Path(stablewm_home).expanduser().resolve()
    return Path(swm.data.utils.get_cache_dir()).resolve()


def stack_actions_by_frameskip(
    action: np.ndarray,
    episode_ids: np.ndarray | None,
    step_idx: np.ndarray | None,
    *,
    frameskip: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if frameskip <= 1:
        return action, episode_ids, step_idx

    starts, ends = contiguous_spans(action.shape[0], episode_ids, step_idx)
    stacked_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []

    for start, end in zip(starts.tolist(), ends.tolist()):
        span = action[start:end]
        span_len = span.shape[0]
        if span_len < frameskip:
            continue

        out_len = span_len - frameskip + 1
        slices = [span[offset : offset + out_len] for offset in range(frameskip)]
        stacked = np.concatenate(slices, axis=1)
        stacked_parts.append(stacked)

        if episode_ids is not None:
            episode_parts.append(episode_ids[start : start + out_len])
        if step_idx is not None:
            step_parts.append(step_idx[start : start + out_len])

    if not stacked_parts:
        raise ValueError(
            f"No valid action windows found for frameskip={frameskip}. "
            "Check episode boundaries and step indices in dataset metadata."
        )

    action_out = np.concatenate(stacked_parts, axis=0).astype(np.float32, copy=False)
    episode_out = np.concatenate(episode_parts, axis=0) if episode_parts else None
    step_out = np.concatenate(step_parts, axis=0) if step_parts else None
    return action_out, episode_out, step_out


def maybe_load_model_object(path: Path) -> torch.nn.Module:
    baseline_root = str(_baseline_adapter.BASELINE_ROOT)
    if baseline_root not in sys.path:
        sys.path.insert(0, baseline_root)

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    model = loaded.model if hasattr(loaded, "model") else loaded
    if not hasattr(model, "latent_action_encoder"):
        raise ValueError(
            f"Checkpoint does not expose latent_action_encoder: {path}"
        )
    model = model.to(torch.device("cpu"))
    model.eval()
    return model


def find_candidate_runs_roots(cli_runs_root: str, cache_dir: Path) -> list[Path]:
    roots: list[Path] = []
    if cli_runs_root:
        roots.append(Path(cli_runs_root).expanduser().resolve())
    roots.append((cache_dir / "runs").resolve())

    scratch_root = Path(f"/scratch-shared/{os.getenv('USER', '')}/stablewm_data/runs")
    if str(scratch_root) != "/scratch-shared//stablewm_data/runs":
        roots.append(scratch_root)

    deduped = []
    seen = set()
    for path in roots:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def resolve_checkpoint(
    *,
    checkpoint: str,
    run_name: str,
    checkpoint_epoch: int,
    runs_root: str,
    cache_dir: Path,
) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    roots = find_candidate_runs_roots(runs_root, cache_dir)
    pattern = (
        f"{run_name}_epoch_{checkpoint_epoch}_object.ckpt"
        if run_name
        else f"*_epoch_{checkpoint_epoch}_object.ckpt"
    )

    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(root.glob(f"**/{pattern}"))

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint candidates found. Provide --checkpoint or set "
            "--run-name/--runs-root to a valid run."
        )

    candidates = sorted(
        (path.resolve() for path in candidates),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0]


def load_action_matrix(
    dataset_name: str,
    cache_dir: Path,
    *,
    expected_action_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=["action", "state", "proprio"],
        cache_dir=cache_dir,
    )
    action = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    if action.ndim != 2 or action.shape[0] == 0:
        raise ValueError(f"Unexpected action matrix shape: {action.shape}")

    episode_ids = None
    for candidate in ("episode_idx", "ep_idx"):
        if candidate in set(dataset.column_names or []):
            episode_ids = np.asarray(dataset.get_col_data(candidate))
            break

    step_idx = None
    if "step_idx" in set(dataset.column_names or []):
        step_idx = np.asarray(dataset.get_col_data("step_idx"))

    valid_mask = ~np.isnan(action).any(axis=1)
    action = action[valid_mask]
    if episode_ids is not None:
        episode_ids = episode_ids[valid_mask]
    if step_idx is not None:
        step_idx = step_idx[valid_mask]

    inferred_frameskip = 1
    if expected_action_dim is not None and action.shape[1] != int(expected_action_dim):
        raw_dim = int(action.shape[1])
        target_dim = int(expected_action_dim)
        if target_dim % raw_dim != 0:
            raise ValueError(
                "Action dimension mismatch between dataset and latent action encoder: "
                f"dataset action dim={raw_dim}, model input dim={target_dim}. "
                "Model input dim is not an integer multiple of dataset action dim."
            )
        inferred_frameskip = target_dim // raw_dim
        action, episode_ids, step_idx = stack_actions_by_frameskip(
            action,
            episode_ids,
            step_idx,
            frameskip=inferred_frameskip,
        )

    # Match training-time normalization behavior for the action column.
    action_t = torch.from_numpy(action)
    mean = action_t.mean(dim=0, keepdim=True)
    std = action_t.std(dim=0, keepdim=True)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    action = ((action_t - mean) / std).cpu().numpy().astype(np.float32)

    return action, episode_ids, step_idx, inferred_frameskip


def contiguous_spans(
    num_rows: int,
    episode_ids: np.ndarray | None,
    step_idx: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if num_rows <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    if episode_ids is None:
        return np.array([0], dtype=np.int64), np.array([num_rows], dtype=np.int64)

    invalid = episode_ids[1:] != episode_ids[:-1]
    if step_idx is not None:
        invalid = np.logical_or(invalid, (step_idx[1:] - step_idx[:-1]) != 1)

    boundaries = np.nonzero(invalid)[0] + 1
    starts = np.concatenate(([0], boundaries)).astype(np.int64)
    ends = np.concatenate((boundaries, [num_rows])).astype(np.int64)
    return starts, ends


def sample_chunks(
    action: np.ndarray,
    span_starts: np.ndarray,
    span_ends: np.ndarray,
    *,
    num_chunks: int,
    min_chunk_len: int,
    max_chunk_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")

    span_lengths = span_ends - span_starts
    if span_lengths.size == 0:
        raise ValueError("No contiguous spans available for chunk sampling.")

    max_span = int(span_lengths.max())
    lo = max(1, int(min_chunk_len))
    hi = min(int(max_chunk_len), max_span)
    if hi < lo:
        raise ValueError(
            f"Invalid chunk-length range after clipping: [{lo}, {hi}] with max_span={max_span}"
        )

    eligible_by_len: dict[int, np.ndarray] = {}
    valid_lengths: list[int] = []
    for chunk_len in range(lo, hi + 1):
        eligible = np.nonzero(span_lengths >= chunk_len)[0]
        if eligible.size > 0:
            eligible_by_len[chunk_len] = eligible
            valid_lengths.append(chunk_len)

    if not valid_lengths:
        raise ValueError("No valid chunk lengths available for sampling.")

    sampled_lengths = rng.choice(np.asarray(valid_lengths), size=num_chunks, replace=True)
    max_len = int(sampled_lengths.max())
    chunks = np.zeros((num_chunks, max_len, action.shape[1]), dtype=np.float32)
    mask = np.zeros((num_chunks, max_len), dtype=bool)

    for i, chunk_len in enumerate(sampled_lengths):
        eligible_spans = eligible_by_len[int(chunk_len)]
        span_id = int(rng.choice(eligible_spans))
        span_start = int(span_starts[span_id])
        span_len = int(span_lengths[span_id])
        offset = int(rng.integers(0, span_len - int(chunk_len) + 1))
        start = span_start + offset
        end = start + int(chunk_len)
        chunks[i, : int(chunk_len)] = action[start:end]
        mask[i, : int(chunk_len)] = True

    return chunks, mask, sampled_lengths.astype(np.int64)


def batched_ranges(total: int, batch_size: int) -> Iterable[tuple[int, int]]:
    cursor = 0
    while cursor < total:
        nxt = min(cursor + batch_size, total)
        yield cursor, nxt
        cursor = nxt


def encode_macro_actions(
    model: torch.nn.Module,
    chunks: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    encode_fn = (
        model.encode_macro_actions
        if hasattr(model, "encode_macro_actions")
        else model.latent_action_encoder
    )

    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for lo, hi in batched_ranges(chunks.shape[0], batch_size):
            chunk_t = torch.from_numpy(chunks[lo:hi]).to(torch.device("cpu"))
            mask_t = torch.from_numpy(mask[lo:hi]).to(torch.device("cpu"))
            latent = encode_fn(chunk_t, mask_t)
            outputs.append(latent.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(outputs, axis=0)


def fit_pca(latent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if latent.ndim != 2:
        raise ValueError("latent must be shape (N, D)")
    centered = latent - latent.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if centered.shape[0] <= 1:
        var = np.zeros((centered.shape[1],), dtype=np.float64)
    else:
        var = (singular**2) / (centered.shape[0] - 1)
    if var.sum() <= 0:
        explained = np.zeros_like(var)
    else:
        explained = var / var.sum()
    return centered, explained, vh


def sample_pairwise_cosine(
    latent: np.ndarray,
    *,
    num_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = latent.shape[0]
    if n < 2 or num_pairs <= 0:
        return np.zeros((0,), dtype=np.float32)
    safe_norm = np.clip(np.linalg.norm(latent, axis=1, keepdims=True), 1e-8, None)
    latent_normed = latent / safe_norm

    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    keep = i != j
    if not np.any(keep):
        return np.zeros((0,), dtype=np.float32)
    i = i[keep]
    j = j[keep]
    cosine = np.sum(latent_normed[i] * latent_normed[j], axis=1)
    return cosine.astype(np.float32)


def interpret_stats(stats: AnalysisStats) -> list[str]:
    lines: list[str] = []

    if stats.pc1_explained_ratio >= 0.40:
        lines.append(
            f"PC1 explains {stats.pc1_explained_ratio:.1%}, indicating a strongly concentrated latent manifold."
        )
    elif stats.pc1_explained_ratio >= 0.25:
        lines.append(
            f"PC1 explains {stats.pc1_explained_ratio:.1%}, showing moderate anisotropy in latent usage."
        )
    else:
        lines.append(
            f"PC1 explains {stats.pc1_explained_ratio:.1%}, which suggests no single dominant latent direction."
        )

    if stats.effective_rank_fraction < 0.30:
        lines.append(
            f"Effective rank is {stats.effective_rank:.2f} ({stats.effective_rank_fraction:.1%} of dimensions), a collapse risk signal."
        )
    elif stats.effective_rank_fraction < 0.55:
        lines.append(
            f"Effective rank is {stats.effective_rank:.2f} ({stats.effective_rank_fraction:.1%} of dimensions), indicating partial dimensional underuse."
        )
    else:
        lines.append(
            f"Effective rank is {stats.effective_rank:.2f} ({stats.effective_rank_fraction:.1%} of dimensions), indicating broad dimension usage."
        )

    corr_abs = abs(stats.length_norm_corr)
    if math.isnan(corr_abs):
        lines.append("Length-to-norm correlation is undefined (single chunk length sampled).")
    elif corr_abs >= 0.30:
        lines.append(
            f"Chunk length and latent norm correlate at {stats.length_norm_corr:.3f}; latent magnitude strongly depends on chunk duration."
        )
    elif corr_abs >= 0.15:
        lines.append(
            f"Chunk length and latent norm correlate at {stats.length_norm_corr:.3f}; dependence exists but is moderate."
        )
    else:
        lines.append(
            f"Chunk length and latent norm correlate at {stats.length_norm_corr:.3f}; latent scale is mostly length-invariant."
        )

    if stats.cosine_mean > 0.30:
        lines.append(
            f"Mean pairwise cosine is {stats.cosine_mean:.3f}, suggesting many macro-actions align in similar directions."
        )
    elif stats.cosine_mean < -0.10:
        lines.append(
            f"Mean pairwise cosine is {stats.cosine_mean:.3f}, indicating anti-correlated latent directions."
        )
    else:
        lines.append(
            f"Mean pairwise cosine is {stats.cosine_mean:.3f}, consistent with a mixed-direction latent cloud."
        )

    return lines


def write_markdown_summary(
    out_dir: Path,
    stats: AnalysisStats,
    interpretations: list[str],
) -> None:
    summary_path = out_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Latent Action Space Analysis\n\n")
        f.write(f"- checkpoint: `{stats.checkpoint}`\n")
        f.write(f"- dataset: `{stats.dataset}`\n")
        f.write(f"- cache_dir: `{stats.cache_dir}`\n")
        f.write(f"- num_chunks: `{stats.num_chunks}`\n")
        f.write(f"- latent_dim: `{stats.latent_dim}`\n")
        f.write(
            f"- chunk_len_range: `{stats.min_chunk_len}` to `{stats.max_chunk_len}` "
            f"(mean `{stats.mean_chunk_len:.2f}`)\n\n"
        )
        f.write("## Key Metrics\n\n")
        f.write(f"- norm mean/std: `{stats.norm_mean:.4f}` / `{stats.norm_std:.4f}`\n")
        f.write(
            f"- norm percentiles p05/p50/p95: `{stats.norm_p05:.4f}` / "
            f"`{stats.norm_p50:.4f}` / `{stats.norm_p95:.4f}`\n"
        )
        f.write(f"- length-norm correlation: `{stats.length_norm_corr:.4f}`\n")
        f.write(
            f"- PC1/PC2 explained variance: `{stats.pc1_explained_ratio:.4f}` / "
            f"`{stats.pc2_explained_ratio:.4f}`\n"
        )
        f.write(
            f"- effective rank (absolute/fraction): `{stats.effective_rank:.4f}` / "
            f"`{stats.effective_rank_fraction:.4f}`\n"
        )
        f.write(f"- isotropy ratio (min/max var): `{stats.isotropy_ratio:.6f}`\n")
        f.write(
            f"- pairwise cosine mean/std: `{stats.cosine_mean:.4f}` / `{stats.cosine_std:.4f}`\n"
        )
        f.write(
            f"- pairwise cosine p05/p95: `{stats.cosine_p05:.4f}` / `{stats.cosine_p95:.4f}`\n\n"
        )
        f.write("## Interpretation\n\n")
        for line in interpretations:
            f.write(f"- {line}\n")


def save_plots(
    out_dir: Path,
    latent: np.ndarray,
    chunk_lengths: np.ndarray,
    explained: np.ndarray,
    components: np.ndarray,
) -> None:
    centered = latent - latent.mean(axis=0, keepdims=True)
    pca_2d = centered @ components[:2].T if components.shape[0] >= 2 else np.zeros((latent.shape[0], 2))
    norms = np.linalg.norm(latent, axis=1)

    unique_lengths = np.unique(chunk_lengths)
    len_means = np.array([norms[chunk_lengths == ln].mean() for ln in unique_lengths])
    len_stds = np.array([norms[chunk_lengths == ln].std() for ln in unique_lengths])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax = axes[0, 0]
    sc = ax.scatter(
        pca_2d[:, 0],
        pca_2d[:, 1],
        c=chunk_lengths,
        s=5,
        alpha=0.55,
        cmap="viridis",
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax, label="chunk length")
    ax.set_title("PCA Projection of Latent Macro-Actions")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax = axes[0, 1]
    ax.hist(norms, bins=60, color="#4c72b0", alpha=0.9)
    ax.set_title("Latent Norm Distribution")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("count")

    ax = axes[1, 0]
    top_k = min(32, explained.shape[0])
    ax.plot(np.arange(1, top_k + 1), explained[:top_k], marker="o", markersize=3)
    ax.set_title("Explained Variance by Principal Component")
    ax.set_xlabel("principal component index")
    ax.set_ylabel("explained variance ratio")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.errorbar(unique_lengths, len_means, yerr=len_stds, fmt="-o", markersize=4)
    ax.set_title("Latent Norm vs Action Chunk Length")
    ax.set_xlabel("chunk length")
    ax.set_ylabel("norm mean ± std")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "latent_action_space_overview.png", dpi=180)
    plt.close(fig)

    # Correlation map (first 32 dims for readability).
    corr = np.corrcoef(latent, rowvar=False)
    k = min(32, corr.shape[0])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr[:k, :k], vmin=-1.0, vmax=1.0, cmap="coolwarm")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Latent Dimension Correlation (first {k} dims)")
    ax.set_xlabel("dimension")
    ax.set_ylabel("dimension")
    fig.tight_layout()
    fig.savefig(out_dir / "latent_action_corr_heatmap.png", dpi=180)
    plt.close(fig)


def build_stats(
    *,
    checkpoint: Path,
    dataset_name: str,
    cache_dir: Path,
    latent: np.ndarray,
    chunk_lengths: np.ndarray,
    explained: np.ndarray,
    cosine: np.ndarray,
) -> AnalysisStats:
    norms = np.linalg.norm(latent, axis=1)
    var = explained

    pc1 = float(var[0]) if var.size > 0 else 0.0
    pc2 = float(var[1]) if var.size > 1 else 0.0
    var_safe = var[var > 0]
    eff_rank = float(np.exp(-(var_safe * np.log(var_safe)).sum())) if var_safe.size > 0 else 0.0
    eff_rank_frac = eff_rank / float(latent.shape[1]) if latent.shape[1] > 0 else 0.0

    # Isotropy proxy: min/max explained variance among non-zero modes.
    isotropy = 0.0
    if var_safe.size > 1:
        isotropy = float(var_safe[-1] / max(var_safe[0], 1e-12))

    if np.unique(chunk_lengths).size > 1:
        length_norm_corr = float(np.corrcoef(chunk_lengths.astype(np.float64), norms)[0, 1])
    else:
        length_norm_corr = float("nan")

    if cosine.size > 0:
        cosine_mean = float(np.mean(cosine))
        cosine_std = float(np.std(cosine))
        cosine_p05 = float(np.percentile(cosine, 5))
        cosine_p95 = float(np.percentile(cosine, 95))
    else:
        cosine_mean = 0.0
        cosine_std = 0.0
        cosine_p05 = 0.0
        cosine_p95 = 0.0

    return AnalysisStats(
        checkpoint=str(checkpoint),
        dataset=dataset_name,
        cache_dir=str(cache_dir),
        num_chunks=int(latent.shape[0]),
        latent_dim=int(latent.shape[1]),
        min_chunk_len=int(chunk_lengths.min()),
        max_chunk_len=int(chunk_lengths.max()),
        mean_chunk_len=float(chunk_lengths.mean()),
        norm_mean=float(norms.mean()),
        norm_std=float(norms.std()),
        norm_p05=float(np.percentile(norms, 5)),
        norm_p50=float(np.percentile(norms, 50)),
        norm_p95=float(np.percentile(norms, 95)),
        length_norm_corr=length_norm_corr,
        pc1_explained_ratio=pc1,
        pc2_explained_ratio=pc2,
        effective_rank=eff_rank,
        effective_rank_fraction=eff_rank_frac,
        isotropy_ratio=isotropy,
        cosine_mean=cosine_mean,
        cosine_std=cosine_std,
        cosine_p05=cosine_p05,
        cosine_p95=cosine_p95,
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    repo_root = resolve_repo_root()
    cache_dir = resolve_cache_dir(args.cache_dir)

    ckpt_path = resolve_checkpoint(
        checkpoint=args.checkpoint,
        run_name=args.run_name,
        checkpoint_epoch=args.checkpoint_epoch,
        runs_root=args.runs_root,
        cache_dir=cache_dir,
    )

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (repo_root / "outputs" / "latent_analysis" / ckpt_path.stem).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[latent-analysis] repo_root: {repo_root}")
    print(f"[latent-analysis] cache_dir: {cache_dir}")
    print(f"[latent-analysis] checkpoint: {ckpt_path}")
    print(f"[latent-analysis] output_dir: {out_dir}")

    model = maybe_load_model_object(ckpt_path)
    model_max_seq_len = int(getattr(model.latent_action_encoder, "max_seq_len", args.max_chunk_len))
    action_expected_dim = int(model.latent_action_encoder.input_proj.in_features)

    action, episode_ids, step_idx, inferred_frameskip = load_action_matrix(
        args.dataset_name,
        cache_dir,
        expected_action_dim=action_expected_dim,
    )
    print(f"[latent-analysis] inferred_frameskip_for_actions: {inferred_frameskip}")
    if action.shape[1] != action_expected_dim:
        raise ValueError(
            "Action dimension mismatch between dataset and latent action encoder: "
            f"dataset action dim={action.shape[1]}, model input dim={action_expected_dim}. "
            "Use a checkpoint/dataset pair from the same training setup."
        )

    starts, ends = contiguous_spans(action.shape[0], episode_ids, step_idx)
    chunks, mask, chunk_lengths = sample_chunks(
        action,
        starts,
        ends,
        num_chunks=args.num_chunks,
        min_chunk_len=args.min_chunk_len,
        max_chunk_len=min(args.max_chunk_len, model_max_seq_len),
        rng=rng,
    )
    latent = encode_macro_actions(model, chunks, mask, batch_size=args.batch_size)

    centered, explained, components = fit_pca(latent)
    _ = centered  # centered is used implicitly in projections built from components.
    cosine = sample_pairwise_cosine(
        latent,
        num_pairs=args.num_cosine_pairs,
        rng=rng,
    )

    stats = build_stats(
        checkpoint=ckpt_path,
        dataset_name=args.dataset_name,
        cache_dir=cache_dir,
        latent=latent,
        chunk_lengths=chunk_lengths,
        explained=explained,
        cosine=cosine,
    )

    save_plots(out_dir, latent, chunk_lengths, explained, components)
    interpretations = interpret_stats(stats)
    write_markdown_summary(out_dir, stats, interpretations)

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f, indent=2, sort_keys=True)

    np.savez_compressed(
        out_dir / "latent_samples.npz",
        latent=latent.astype(np.float32),
        chunk_lengths=chunk_lengths.astype(np.int64),
        explained_variance=explained.astype(np.float64),
    )

    print("[latent-analysis] completed.")
    print(f"[latent-analysis] stats: {out_dir / 'stats.json'}")
    print(f"[latent-analysis] summary: {out_dir / 'summary.md'}")
    print(f"[latent-analysis] plots: {out_dir / 'latent_action_space_overview.png'}")


if __name__ == "__main__":
    main()
