from __future__ import annotations

import math
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from h_le_wm.probe.model import (
    LatentToPixelDecoder,
    compute_psnr,
    denormalize_imagenet,
    infer_latent_dim,
    load_decoder_state_dict,
    load_hi_checkpoint,
)
from h_le_wm.train.hierarchical import build_action_chunks_batched
from h_le_wm.probe.train import build_dataset_and_loaders


DEFAULT_HI_CKPT = (
    "/scratch-shared/scur0200/stablewm_data/runs/"
    "hi_lewm_p2_train_hope2_22253175/"
    "hi_lewm_p2_train_hope2_22253175_epoch_15_object.ckpt"
)
DEFAULT_ENV_DUMP = REPO_ROOT / "environment.json"
DEFAULT_SHARED_CACHE = Path(f"/scratch-shared/{os.environ.get('USER', 'scur0200')}/stablewm_data")


@dataclass(slots=True)
class ProbeBundle:
    cfg: object
    hi_model: torch.nn.Module
    decoder: torch.nn.Module
    train_loader: object
    val_loader: object
    run_dir: Path
    probe_ckpt: Path
    device: torch.device


def load_environment_dump(path: str | Path = DEFAULT_ENV_DUMP) -> dict:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Environment dump does not exist: {path}")
    with path.open("r") as f:
        return json.load(f)


def summarize_environment_dump(env: dict) -> pd.Series:
    python_info = env.get("python", {})
    system_info = env.get("system", {})
    cuda_info = env.get("cuda", {})
    git_info = env.get("git", {})
    slurm_info = env.get("slurm", {})
    packages = env.get("packages", [])
    return pd.Series(
        {
            "timestamp": env.get("timestamp"),
            "python_version": python_info.get("version"),
            "hostname": system_info.get("hostname"),
            "platform": system_info.get("platform"),
            "cuda_available": cuda_info.get("available"),
            "cuda_device_count": cuda_info.get("device_count"),
            "git_commit": git_info.get("commit"),
            "slurm_job_id": slurm_info.get("job_id"),
            "slurm_job_name": slurm_info.get("job_name"),
            "num_packages": len(packages) if isinstance(packages, list) else None,
        }
    )


def load_probe_cfg(cfg_path: str | Path):
    cfg = OmegaConf.load(Path(cfg_path))
    return cfg


def clone_cfg(cfg):
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def resolve_run_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {path}")
    return path


def infer_cache_dir(*, run_dir: str | Path | None = None, hi_ckpt: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if run_dir is not None:
        run_path = Path(run_dir).expanduser().resolve()
        candidates.extend([run_path, *run_path.parents])
    if hi_ckpt is not None:
        ckpt_path = Path(hi_ckpt).expanduser().resolve()
        candidates.extend([ckpt_path.parent, *ckpt_path.parents])
    candidates.append(DEFAULT_SHARED_CACHE)

    for candidate in candidates:
        if candidate.name == "runs" and candidate.parent.exists():
            return candidate.parent
        if candidate.exists() and (candidate / "pusht_expert_train.h5").exists():
            return candidate

    raise FileNotFoundError(
        "Unable to infer StableWM cache dir. Set STABLEWM_HOME manually or pass a run/checkpoint "
        "under the shared stablewm_data tree."
    )


def find_latest_probe_checkpoint(run_dir: str | Path) -> Path:
    run_dir = resolve_run_dir(run_dir)
    candidates = sorted(run_dir.glob("*_probe.pt"))
    if not candidates:
        raise FileNotFoundError(f"No probe checkpoint found under: {run_dir}")
    latest = [p for p in candidates if "_epoch_" not in p.name]
    return latest[0] if latest else candidates[-1]


def find_epoch_probe_checkpoints(run_dir: str | Path) -> list[Path]:
    run_dir = resolve_run_dir(run_dir)
    return sorted(run_dir.glob("*_epoch_*_probe.pt"))


def load_probe_bundle(
    *,
    run_dir: str | Path,
    probe_ckpt: str | Path | None = None,
    hi_ckpt: str | Path = DEFAULT_HI_CKPT,
    split: str = "val",
    cache_dir: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    device: str | torch.device | None = None,
):
    run_dir = resolve_run_dir(run_dir)
    cfg = load_probe_cfg(run_dir / "config.yaml")
    cfg = clone_cfg(cfg)
    cfg.probe.checkpoint.path = str(hi_ckpt)
    if batch_size is not None:
        cfg.loader.batch_size = int(batch_size)
    if num_workers is not None:
        cfg.loader.num_workers = int(num_workers)
    if pin_memory is not None:
        cfg.loader.pin_memory = bool(pin_memory)
    if persistent_workers is not None:
        cfg.loader.persistent_workers = bool(persistent_workers)
    if prefetch_factor is not None:
        cfg.loader.prefetch_factor = int(prefetch_factor)
    if int(cfg.loader.num_workers) <= 0:
        cfg.loader.num_workers = 0
        cfg.loader.persistent_workers = False
        cfg.loader.pin_memory = False
        if 'prefetch_factor' in cfg.loader:
            del cfg.loader['prefetch_factor']
    resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir else infer_cache_dir(run_dir=run_dir, hi_ckpt=hi_ckpt)
    os.environ["STABLEWM_HOME"] = str(resolved_cache_dir)
    probe_ckpt_path = Path(probe_ckpt).expanduser() if probe_ckpt else find_latest_probe_checkpoint(run_dir)

    hi_model = load_hi_checkpoint(cfg.probe.checkpoint.path)
    latent_dim = infer_latent_dim(hi_model)
    decoder_cfg = OmegaConf.to_container(cfg.probe.decoder, resolve=True)
    decoder = LatentToPixelDecoder(latent_dim=latent_dim, img_size=int(cfg.img_size), **decoder_cfg)
    decoder.load_state_dict(load_decoder_state_dict(probe_ckpt_path), strict=True)

    runtime_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hi_model = hi_model.to(runtime_device).eval()
    decoder = decoder.to(runtime_device).eval()

    train_loader, val_loader = build_dataset_and_loaders(cfg)
    return ProbeBundle(
        cfg=cfg,
        hi_model=hi_model,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=run_dir,
        probe_ckpt=probe_ckpt_path,
        device=runtime_device,
    )


def _get_loader(bundle: ProbeBundle, split: str):
    if split == "train":
        return bundle.train_loader
    if split == "val":
        return bundle.val_loader
    raise ValueError("split must be 'train' or 'val'")


def sample_batch(bundle: ProbeBundle, *, split: str = "val", batch_index: int = 0):
    loader = _get_loader(bundle, split)
    for idx, batch in enumerate(loader):
        if idx == batch_index:
            return batch
    raise IndexError(f"Requested batch_index={batch_index}, but loader is shorter.")


@torch.inference_mode()
def run_decoder_probe(bundle: ProbeBundle, batch: dict):
    device = bundle.device
    waypoints = batch["waypoints"].to(device=device, dtype=torch.long)
    pixels = batch["pixels"].to(device=device)
    actions = torch.nan_to_num(batch["action"].to(device=device), 0.0)

    encoded = bundle.hi_model.encode({"pixels": pixels}, encode_actions=False)
    z_waypoints = encoded["emb"]
    z_target = z_waypoints[:, 1:]

    starts = waypoints[:, :-1]
    ends = waypoints[:, 1:]
    chunk_actions, chunk_mask = build_action_chunks_batched(actions, starts, ends)
    b, k, l_max, act_dim = chunk_actions.shape
    flat_actions = chunk_actions.reshape(b * k, l_max, act_dim)
    flat_mask = chunk_mask.reshape(b * k, l_max)
    flat_macro = bundle.hi_model.encode_macro_actions(flat_actions, flat_mask)
    z_pred = bundle.hi_model.predict_high(z_waypoints[:, :-1], flat_macro.reshape(b, k, -1))

    context = pixels[:, :-1].reshape(-1, *pixels.shape[2:])
    target = pixels[:, 1:].reshape(-1, *pixels.shape[2:])
    flat_z_target = z_target.reshape(-1, z_target.size(-1))
    flat_z_pred = z_pred.reshape(-1, z_pred.size(-1))
    decoded_true = bundle.decoder(flat_z_target)
    decoded_pred = bundle.decoder(flat_z_pred)

    target_vis = denormalize_imagenet(target).detach().cpu()
    context_vis = denormalize_imagenet(context).detach().cpu()
    decoded_true_vis = denormalize_imagenet(decoded_true).detach().cpu()
    decoded_pred_vis = denormalize_imagenet(decoded_pred).detach().cpu()
    error_map = (decoded_pred_vis - target_vis).abs().mean(dim=1)

    metrics = {
        "pixel_mse_true": float(F.mse_loss(decoded_true, target).detach().cpu().item()),
        "pixel_mse_pred": float(F.mse_loss(decoded_pred, target).detach().cpu().item()),
        "psnr_true": float(compute_psnr(decoded_true_vis, target_vis).detach().cpu().item()),
        "psnr_pred": float(compute_psnr(decoded_pred_vis, target_vis).detach().cpu().item()),
        "latent_gap": float(F.mse_loss(flat_z_pred, flat_z_target).detach().cpu().item()),
    }

    return {
        "context_vis": context_vis,
        "target_vis": target_vis,
        "decoded_true_vis": decoded_true_vis,
        "decoded_pred_vis": decoded_pred_vis,
        "error_map": error_map,
        "metrics": metrics,
        "num_transitions": k,
        "batch_size": b,
        "waypoints": waypoints.detach().cpu(),
    }


@torch.inference_mode()
def evaluate_bundle(bundle: ProbeBundle, *, split: str = "val", max_batches: int = 8) -> pd.DataFrame:
    loader = _get_loader(bundle, split)
    rows = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        outputs = run_decoder_probe(bundle, batch)
        row = {"batch_index": idx, **outputs["metrics"]}
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame) -> pd.Series:
    summary = df.mean(numeric_only=True)
    summary["num_batches"] = float(len(df))
    return summary


def _hide_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_waypoint_panel(
    outputs: dict,
    *,
    save_path: str | Path,
    num_rows: int = 4,
    title: str = "Decoder Probe: Context, Target, Decoded True, Decoded Pred",
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(num_rows, outputs["target_vis"].shape[0])
    fig, axes = plt.subplots(n, 5, figsize=(14, 2.6 * n), constrained_layout=True)
    axes = np.atleast_2d(axes)
    col_titles = ["Context", "Target", "Decoded True", "Decoded Pred", "Abs Error"]
    for col, label in enumerate(col_titles):
        axes[0, col].set_title(label, fontsize=12, fontweight="bold")

    for row in range(n):
        images = [
            outputs["context_vis"][row].permute(1, 2, 0).numpy(),
            outputs["target_vis"][row].permute(1, 2, 0).numpy(),
            outputs["decoded_true_vis"][row].permute(1, 2, 0).numpy(),
            outputs["decoded_pred_vis"][row].permute(1, 2, 0).numpy(),
        ]
        for col, image in enumerate(images):
            axes[row, col].imshow(np.clip(image, 0.0, 1.0))
            _hide_axes(axes[row, col])
        axes[row, 4].imshow(outputs["error_map"][row].numpy(), cmap="magma")
        _hide_axes(axes[row, 4])

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_rollout_story_figure(
    outputs: dict,
    *,
    save_path: str | Path,
    sample_index: int = 0,
    title: str = "Waypoint Rollout Story",
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    transitions = outputs["num_transitions"]
    idx0 = sample_index * transitions
    idxs = [idx0 + t for t in range(transitions)]

    fig, axes = plt.subplots(4, transitions, figsize=(3.2 * transitions, 10), constrained_layout=True)
    row_titles = ["Context", "Target", "Decoded True", "Decoded Pred"]
    for row, row_title in enumerate(row_titles):
        axes[row, 0].set_ylabel(row_title, fontsize=12, fontweight="bold")

    for col, flat_idx in enumerate(idxs):
        images = [
            outputs["context_vis"][flat_idx].permute(1, 2, 0).numpy(),
            outputs["target_vis"][flat_idx].permute(1, 2, 0).numpy(),
            outputs["decoded_true_vis"][flat_idx].permute(1, 2, 0).numpy(),
            outputs["decoded_pred_vis"][flat_idx].permute(1, 2, 0).numpy(),
        ]
        axes[0, col].set_title(f"Step {col + 1}", fontsize=12, fontweight="bold")
        for row, image in enumerate(images):
            axes[row, col].imshow(np.clip(image, 0.0, 1.0))
            _hide_axes(axes[row, col])

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_checkpoint_comparison_figure(
    bundles: Iterable[tuple[str, ProbeBundle]],
    *,
    batch: dict,
    save_path: str | Path,
    sample_index: int = 0,
    title: str = "Decoder Checkpoint Comparison",
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_items = list(bundles)
    outputs_per_bundle = [(label, run_decoder_probe(bundle, batch)) for label, bundle in bundle_items]

    target = outputs_per_bundle[0][1]["target_vis"][sample_index].permute(1, 2, 0).numpy()
    context = outputs_per_bundle[0][1]["context_vis"][sample_index].permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(len(outputs_per_bundle) + 1, 3, figsize=(10, 3.0 * (len(outputs_per_bundle) + 1)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    headers = ["Context", "Target", "Decoded Pred"]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=12, fontweight="bold")
    axes[0, 0].imshow(np.clip(context, 0.0, 1.0))
    axes[0, 1].imshow(np.clip(target, 0.0, 1.0))
    axes[0, 2].axis("off")
    for col in range(3):
        _hide_axes(axes[0, col])

    for row, (label, outputs) in enumerate(outputs_per_bundle, start=1):
        axes[row, 0].text(0.5, 0.5, label, ha="center", va="center", fontsize=12, fontweight="bold")
        _hide_axes(axes[row, 0])
        axes[row, 1].imshow(np.clip(target, 0.0, 1.0))
        _hide_axes(axes[row, 1])
        pred = outputs["decoded_pred_vis"][sample_index].permute(1, 2, 0).numpy()
        axes[row, 2].imshow(np.clip(pred, 0.0, 1.0))
        _hide_axes(axes[row, 2])

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def export_diverse_waypoint_gallery(
    bundle: ProbeBundle,
    *,
    out_dir: str | Path,
    split: str = "val",
    num_examples: int = 10,
    rows_per_panel: int = 1,
    examples_per_batch: int = 2,
    save_story_figures: bool = True,
) -> pd.DataFrame:
    """Export a folder of diverse decoder-probe figures across loader batches.

    Each exported example uses a different sample from the dataset loader.
    For each selected example this writes:

    - a compact waypoint panel
    - optionally a rollout story figure

    Returns a dataframe with file paths and metrics for each exported example.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = _get_loader(bundle, split)
    exported_rows = []
    example_counter = 0

    for batch_index, batch in enumerate(loader):
        outputs = run_decoder_probe(bundle, batch)
        transitions = int(outputs["num_transitions"])
        batch_size = int(outputs["batch_size"])

        for sample_index in range(min(batch_size, examples_per_batch)):
            if example_counter >= num_examples:
                break

            base_name = f"{split}_example_{example_counter:02d}_batch_{batch_index:03d}_sample_{sample_index:02d}"
            panel_path = out_dir / f"{base_name}_panel.png"
            story_path = out_dir / f"{base_name}_story.png"

            start = sample_index * transitions
            stop = start + transitions
            sample_outputs = {
                "context_vis": outputs["context_vis"][start:stop],
                "target_vis": outputs["target_vis"][start:stop],
                "decoded_true_vis": outputs["decoded_true_vis"][start:stop],
                "decoded_pred_vis": outputs["decoded_pred_vis"][start:stop],
                "error_map": outputs["error_map"][start:stop],
                "metrics": outputs["metrics"],
                "num_transitions": transitions,
                "batch_size": 1,
                "waypoints": outputs["waypoints"][sample_index : sample_index + 1],
            }

            save_waypoint_panel(
                sample_outputs,
                save_path=panel_path,
                num_rows=min(rows_per_panel, transitions),
                title=f"{split.title()} Example {example_counter}: Context / Target / Decoded / Error",
            )

            if save_story_figures:
                save_rollout_story_figure(
                    sample_outputs,
                    save_path=story_path,
                    sample_index=0,
                    title=f"{split.title()} Example {example_counter}: Waypoint Rollout Story",
                )

            exported_rows.append(
                {
                    "example_index": example_counter,
                    "batch_index": batch_index,
                    "sample_index": sample_index,
                    "panel_path": str(panel_path),
                    "story_path": str(story_path) if save_story_figures else "",
                    "pixel_mse_true": outputs["metrics"]["pixel_mse_true"],
                    "pixel_mse_pred": outputs["metrics"]["pixel_mse_pred"],
                    "psnr_true": outputs["metrics"]["psnr_true"],
                    "psnr_pred": outputs["metrics"]["psnr_pred"],
                    "latent_gap": outputs["metrics"]["latent_gap"],
                }
            )
            example_counter += 1

        if example_counter >= num_examples:
            break

    df = pd.DataFrame(exported_rows)
    if not df.empty:
        df.to_csv(out_dir / "gallery_index.csv", index=False)
    return df


def make_output_dir(run_dir: str | Path, *, name: str = "notebook_exports") -> Path:
    path = resolve_run_dir(run_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_visuals(run_dir: str | Path) -> list[Path]:
    run_dir = resolve_run_dir(run_dir)
    return sorted((run_dir / "visuals").glob("*.png"))
