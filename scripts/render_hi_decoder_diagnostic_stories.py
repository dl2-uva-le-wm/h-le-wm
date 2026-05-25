#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from h_le_wm.probe.model import LatentToPixelDecoder, denormalize_imagenet, load_decoder_state_dict
from decoder_probe_notebook_utils import find_latest_probe_checkpoint
from hi_diagnostics import get_row_data_safe, partition_total, resolve_cache_dir


plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render decoder-based rollout story figures from saved diagnostics.")
    p.add_argument("--probe-run-dir", type=Path, required=True)
    p.add_argument("--probe-ckpt", type=Path, default=None)
    p.add_argument("--teacher-artifact", type=Path, required=True)
    p.add_argument("--generated-artifact", type=Path, required=True)
    p.add_argument("--online-artifact", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dataset-name", default="pusht_expert_train")
    p.add_argument("--goal-offset-steps", type=int, default=50)
    p.add_argument("--frame-skip", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-teacher-cases", type=int, default=4)
    p.add_argument("--num-generated-cases", type=int, default=6)
    p.add_argument("--num-online-cases", type=int, default=4)
    p.add_argument("--rows-per-figure", type=int, default=2)
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr.astype(np.float32, copy=False)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def load_dataset(*, dataset_name: str, cache_dir: str | None):
    if cache_dir is not None:
        resolved_cache = resolve_cache_dir(cache_dir)
    else:
        shared_cache = Path(f"/scratch-shared/{os.environ.get('USER', 'user')}/stablewm_data")
        resolved_cache = shared_cache if shared_cache.exists() else resolve_cache_dir(None)
    return swm.data.HDF5Dataset(dataset_name, keys_to_cache=["action"], cache_dir=resolved_cache)


def load_probe_cfg(run_dir: Path, probe_ckpt: Path | None):
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)

    ckpt = probe_ckpt if probe_ckpt is not None else find_latest_probe_checkpoint(run_dir)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("cfg") if isinstance(payload, dict) else None
    if cfg is None:
        raise FileNotFoundError(
            f"Missing probe config at {config_path} and no embedded cfg found in {ckpt}."
        )
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def load_decoder(*, run_dir: Path, probe_ckpt: Path | None, latent_dim: int, device: torch.device):
    cfg = load_probe_cfg(run_dir, probe_ckpt)
    ckpt = probe_ckpt if probe_ckpt is not None else find_latest_probe_checkpoint(run_dir)
    decoder_cfg = OmegaConf.to_container(cfg.probe.decoder, resolve=True)
    decoder = LatentToPixelDecoder(
        latent_dim=int(latent_dim),
        img_size=int(cfg.img_size),
        **decoder_cfg,
    )
    decoder.load_state_dict(load_decoder_state_dict(ckpt), strict=True)
    decoder = decoder.to(device).eval()
    return decoder, ckpt


@torch.inference_mode()
def decode_latents(decoder: torch.nn.Module, latents: np.ndarray, device: torch.device, *, batch_size: int = 256) -> np.ndarray:
    arr = np.asarray(latents, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"Expected latent array with ndim >= 2, got shape {arr.shape}")
    flat = arr.reshape(-1, arr.shape[-1])
    outputs = []
    for start in range(0, flat.shape[0], batch_size):
        chunk = torch.from_numpy(flat[start : start + batch_size]).to(device)
        decoded = decoder(chunk)
        vis = denormalize_imagenet(decoded).detach().cpu().permute(0, 2, 3, 1).numpy()
        outputs.append(np.clip(vis, 0.0, 1.0))
    return np.concatenate(outputs, axis=0).reshape(*arr.shape[:-1], *outputs[0].shape[1:])


def fetch_pixels(dataset, row_indices: np.ndarray) -> np.ndarray:
    rows = get_row_data_safe(dataset, np.asarray(row_indices, dtype=np.int64))
    pixels = np.asarray(rows["pixels"])
    return np.stack([normalize_image(px) for px in pixels], axis=0)


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def hide_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_strip(
    images: list[np.ndarray | None],
    titles: list[str],
    *,
    save_path: Path,
    figure_title: str,
    tile_width: float = 2.2,
    tile_height: float = 2.3,
) -> None:
    ncols = len(images)
    fig, axes = plt.subplots(1, ncols, figsize=(tile_width * ncols, tile_height), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, image, title in zip(axes, images, titles, strict=True):
        if image is not None:
            ax.imshow(image)
        ax.set_title(title, fontsize=9, fontweight="bold")
        hide_axes(ax)
    fig.suptitle(figure_title, fontsize=13, fontweight="bold")
    save_figure(fig, save_path)


def draw_strip_grid(
    strip_rows: list[list[np.ndarray | None]],
    title_rows: list[list[str]],
    row_labels: list[str],
    *,
    save_path: Path,
    figure_title: str,
    tile_width: float = 2.2,
    tile_height: float = 2.3,
) -> None:
    nrows = len(strip_rows)
    ncols = max(len(row) for row in strip_rows)
    padded_images = [row + [None] * (ncols - len(row)) for row in strip_rows]
    padded_titles = [row + [""] * (ncols - len(row)) for row in title_rows]
    fig, axes = plt.subplots(nrows, ncols, figsize=(tile_width * ncols, tile_height * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            image = padded_images[row_idx][col_idx]
            title = padded_titles[row_idx][col_idx]
            if image is not None:
                ax.imshow(image)
            if title:
                ax.set_title(title, fontsize=9, fontweight="bold")
            if col_idx == 0 and row_idx < len(row_labels):
                ax.set_ylabel(row_labels[row_idx], fontsize=10, fontweight="bold")
            hide_axes(ax)
    fig.suptitle(figure_title, fontsize=13, fontweight="bold")
    save_figure(fig, save_path)


def choose_top_indices(scores: np.ndarray, n: int) -> list[int]:
    order = np.argsort(scores)[::-1]
    return [int(i) for i in order[: max(0, min(int(n), len(order)))]]


def choose_generated_indices(successes: np.ndarray, offset_error: np.ndarray, final_error: np.ndarray, n: int) -> list[int]:
    scores = np.abs(offset_error) + 0.5 * final_error + (~successes).astype(np.float32) * 10.0
    return choose_top_indices(scores, n)


def batched_indices(items: list[int], batch_size: int) -> list[list[int]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def render_teacher_stories(
    *,
    dataset,
    decoder,
    device: torch.device,
    teacher_npz: Path,
    out_dir: Path,
    goal_offset_steps: int,
    frame_skip: int,
    num_cases: int,
    rows_per_figure: int,
) -> list[dict[str, object]]:
    data = np.load(teacher_npz, allow_pickle=True)
    starts = np.asarray(data["starts"], dtype=np.int64)
    target_seq = np.asarray(data["target_seq"], dtype=np.float32)
    teacher_pred = np.asarray(data["teacher_pred"], dtype=np.float32)
    open_true_pred = np.asarray(data["open_true_pred"], dtype=np.float32)
    open_cem_pred = np.asarray(data["open_cem_pred"], dtype=np.float32)
    teacher_err = np.asarray(data["teacher_err"], dtype=np.float32).mean(axis=1)
    open_true_err = np.asarray(data["open_true_err"], dtype=np.float32).mean(axis=1)
    open_cem_err = np.asarray(data["open_cem_err"], dtype=np.float32).mean(axis=1)

    score = open_cem_err - teacher_err
    chosen = choose_top_indices(score, num_cases)
    horizon = int(target_seq.shape[1])
    spans = partition_total(max(1, int(math.ceil(goal_offset_steps / frame_skip))), horizon)
    cumulative_steps = np.cumsum([span * frame_skip for span in spans]).astype(np.int64)
    context_offsets = np.concatenate(([0], cumulative_steps[:-1]))

    teacher_dec = decode_latents(decoder, teacher_pred[chosen], device)
    open_true_dec = decode_latents(decoder, open_true_pred[chosen], device)
    open_cem_dec = decode_latents(decoder, open_cem_pred[chosen], device)
    meta_rows: list[dict[str, object]] = []

    strip_images: list[list[np.ndarray | None]] = []
    strip_titles: list[list[str]] = []
    strip_labels: list[str] = []
    for local_idx, sample_idx in enumerate(chosen):
        context_rows = starts[sample_idx] + context_offsets
        target_rows = starts[sample_idx] + cumulative_steps
        context_img = fetch_pixels(dataset, context_rows)
        target_img = fetch_pixels(dataset, target_rows)
        images: list[np.ndarray | None] = []
        titles: list[str] = []
        for step in range(horizon):
            images.extend(
                [
                    context_img[step],
                    target_img[step],
                    teacher_dec[local_idx, step],
                    open_true_dec[local_idx, step],
                    open_cem_dec[local_idx, step],
                ]
            )
            titles.extend(
                [
                    f"C{step + 1}",
                    f"T{step + 1}",
                    f"TF {step + 1}",
                    f"OT {step + 1}",
                    f"CEM {step + 1}",
                ]
            )
        strip_images.append(images)
        strip_titles.append(titles)
        strip_labels.append(f"Case {local_idx + 1}\nTF {teacher_err[sample_idx]:.3f}\nCEM {open_cem_err[sample_idx]:.3f}")
        meta_rows.append(
            {
                "figure_group": int(local_idx // rows_per_figure),
                "sample_index": int(sample_idx),
                "teacher_err_mean": float(teacher_err[sample_idx]),
                "open_true_err_mean": float(open_true_err[sample_idx]),
                "open_cem_err_mean": float(open_cem_err[sample_idx]),
            }
        )
    for fig_idx, batch in enumerate(batched_indices(list(range(len(strip_images))), rows_per_figure)):
        out_path = out_dir / f"teacher_story_{fig_idx:02d}.png"
        draw_strip_grid(
            [strip_images[i] for i in batch],
            [strip_titles[i] for i in batch],
            [strip_labels[i] for i in batch],
            save_path=out_path,
            figure_title="Teacher vs Open-Loop",
        )
    return meta_rows


def nearest_future_offsets_per_sample(generated_stage_targets: np.ndarray, future_latents: np.ndarray) -> np.ndarray:
    generated = torch.from_numpy(np.asarray(generated_stage_targets, dtype=np.float32))
    future = torch.from_numpy(np.asarray(future_latents[:, 1:, :], dtype=np.float32))
    offsets: list[np.ndarray] = []
    for stage_idx in range(generated.shape[1]):
        dist = torch.cdist(generated[:, stage_idx : stage_idx + 1, :], future).squeeze(1)
        nearest = torch.argmin(dist, dim=1) + 1
        offsets.append(nearest.detach().cpu().numpy())
    return np.stack(offsets, axis=1)


def render_generated_stories(
    *,
    dataset,
    decoder,
    device: torch.device,
    generated_npz: Path,
    out_dir: Path,
    goal_offset_steps: int,
    frame_skip: int,
    num_cases: int,
    rows_per_figure: int,
) -> list[dict[str, object]]:
    data = np.load(generated_npz, allow_pickle=True)
    sampled_indices = np.asarray(data["sampled_indices"], dtype=np.int64)
    generated_targets = np.asarray(data["generated_stage_targets"], dtype=np.float32)
    future_latents = np.asarray(data["future_latents"], dtype=np.float32)
    final_latent = np.asarray(data["final_latent"], dtype=np.float32)
    goal_latent = np.asarray(data["goal_latent"], dtype=np.float32)
    successes = np.asarray(data["episode_successes"], dtype=bool)
    stage_end_actual_latents = np.asarray(data["stage_end_actual_latents"], dtype=np.float32) if "stage_end_actual_latents" in data.files else np.empty((0,), dtype=np.float32)

    horizon = int(generated_targets.shape[1])
    spans = partition_total(max(1, int(math.ceil(goal_offset_steps / frame_skip))), horizon)
    expected_offsets = np.cumsum([span * frame_skip for span in spans]).astype(np.int64)
    nearest_offsets = nearest_future_offsets_per_sample(generated_targets, future_latents)
    stage1_offset_error = nearest_offsets[:, 0] - expected_offsets[0]
    final_error = np.mean((final_latent - goal_latent) ** 2, axis=1)
    chosen = choose_generated_indices(successes, stage1_offset_error, final_error, num_cases)

    generated_dec = decode_latents(decoder, generated_targets[chosen], device)
    final_dec = decode_latents(decoder, final_latent[chosen], device)
    stage_end_dec = (
        decode_latents(decoder, np.transpose(stage_end_actual_latents[:, chosen, :], (1, 0, 2)), device)
        if stage_end_actual_latents.ndim == 3 and stage_end_actual_latents.shape[0] > 0
        else None
    )

    meta_rows: list[dict[str, object]] = []
    strip_images: list[list[np.ndarray | None]] = []
    strip_titles: list[list[str]] = []
    strip_labels: list[str] = []
    for local_idx, sample_idx in enumerate(chosen):
        ref_rows = np.concatenate(([sampled_indices[sample_idx]], sampled_indices[sample_idx] + expected_offsets))
        reference_imgs = fetch_pixels(dataset, ref_rows)
        nearest_rows = sampled_indices[sample_idx] + nearest_offsets[sample_idx]
        nearest_imgs = fetch_pixels(dataset, nearest_rows)
        images = [reference_imgs[0]]
        titles = ["Start"]
        for stage_idx in range(horizon):
            images.extend(
                [
                    reference_imgs[stage_idx + 1],
                    generated_dec[local_idx, stage_idx],
                    nearest_imgs[stage_idx],
                    stage_end_dec[local_idx, stage_idx] if stage_end_dec is not None and stage_idx < stage_end_dec.shape[1] else None,
                ]
            )
            titles.extend(
                [
                    f"T{stage_idx + 1}",
                    f"Pred {stage_idx + 1}",
                    f"Near {stage_idx + 1}",
                    f"Reach {stage_idx + 1}",
                ]
            )
        images.append(final_dec[local_idx])
        titles.append("Final")
        strip_images.append(images)
        strip_titles.append(titles)
        strip_labels.append(
            f"Case {local_idx + 1}\n{'success' if successes[sample_idx] else 'fail'}\noff {int(stage1_offset_error[sample_idx]):+d}"
        )
        meta_rows.append(
            {
                "figure_group": int(local_idx // rows_per_figure),
                "sample_index": int(sample_idx),
                "success": bool(successes[sample_idx]),
                "stage1_offset_error_steps": int(stage1_offset_error[sample_idx]),
                "final_error_mean": float(final_error[sample_idx]),
            }
        )
    for fig_idx, batch in enumerate(batched_indices(list(range(len(strip_images))), rows_per_figure)):
        out_path = out_dir / f"generated_story_{fig_idx:02d}.png"
        draw_strip_grid(
            [strip_images[i] for i in batch],
            [strip_titles[i] for i in batch],
            [strip_labels[i] for i in batch],
            save_path=out_path,
            figure_title="Generated Subgoal Acting",
        )
    return meta_rows


def render_online_stories(
    *,
    decoder,
    device: torch.device,
    online_npz: Path | None,
    out_dir: Path,
    num_cases: int,
    rows_per_figure: int,
) -> list[dict[str, object]]:
    if online_npz is None or not online_npz.exists():
        return []
    data = np.load(online_npz, allow_pickle=True)
    required = {"high_plan_current_latents", "high_plan_subgoal_latents", "low_block_actual_latents", "low_block_high_plan_ids"}
    if not required.issubset(set(data.files)):
        return []

    high_plan_current = np.asarray(data["high_plan_current_latents"], dtype=np.float32)
    high_plan_subgoal = np.asarray(data["high_plan_subgoal_latents"], dtype=np.float32)
    low_block_actual = np.asarray(data["low_block_actual_latents"], dtype=np.float32)
    low_block_plan_ids = np.asarray(data["low_block_high_plan_ids"], dtype=np.int64)
    final_latent = np.asarray(data["final_latent"], dtype=np.float32)
    goal_latent = np.asarray(data["goal_latent"], dtype=np.float32)
    successes = np.asarray(data["episode_successes"], dtype=bool)
    high_plan_steps = np.asarray(data["high_plan_steps"], dtype=np.int64) if "high_plan_steps" in data.files else None

    if high_plan_current.ndim != 3 or high_plan_current.shape[0] == 0:
        return []

    high_plan_current_dec = decode_latents(decoder, np.transpose(high_plan_current, (1, 0, 2)), device)
    high_plan_subgoal_dec = decode_latents(decoder, np.transpose(high_plan_subgoal, (1, 0, 2)), device)
    final_dec = decode_latents(decoder, final_latent, device)
    goal_dec = decode_latents(decoder, goal_latent, device)
    low_block_actual_dec = (
        decode_latents(decoder, np.transpose(low_block_actual, (1, 0, 2)), device)
        if low_block_actual.ndim == 3 and low_block_actual.shape[0] > 0
        else None
    )

    final_error = np.mean((final_latent - goal_latent) ** 2, axis=1)
    score = final_error + (~successes).astype(np.float32) * 10.0
    chosen = choose_top_indices(score, num_cases)
    meta_rows: list[dict[str, object]] = []
    max_plans_to_show = min(4, int(high_plan_current_dec.shape[1]))

    strip_images: list[list[np.ndarray | None]] = []
    strip_titles: list[list[str]] = []
    strip_labels: list[str] = []
    for local_idx, sample_idx in enumerate(chosen):
        num_plans = int(high_plan_current_dec.shape[1])
        images = []
        titles = []
        for plan_idx in range(max_plans_to_show):
            reached = None
            if low_block_actual_dec is not None:
                match = np.where(low_block_plan_ids == plan_idx)[0]
                if match.size > 0 and match[0] < low_block_actual_dec.shape[1]:
                    reached = low_block_actual_dec[sample_idx, match[0]]
            images.extend(
                [
                    high_plan_current_dec[sample_idx, plan_idx],
                    high_plan_subgoal_dec[sample_idx, plan_idx],
                    reached,
                ]
            )
            step_label = f"@{int(high_plan_steps[plan_idx])}" if high_plan_steps is not None and plan_idx < len(high_plan_steps) else f"{plan_idx + 1}"
            titles.extend(
                [
                    f"C{plan_idx + 1}{step_label}",
                    f"S{plan_idx + 1}",
                    f"R{plan_idx + 1}",
                ]
            )
        images.extend([goal_dec[sample_idx], final_dec[sample_idx]])
        titles.extend(["Goal", "Final"])
        strip_images.append(images)
        strip_titles.append(titles)
        strip_labels.append(f"Case {local_idx + 1}\n{'success' if successes[sample_idx] else 'fail'}\n{num_plans} replans")
        meta_rows.append(
            {
                "figure_group": int(local_idx // rows_per_figure),
                "sample_index": int(sample_idx),
                "success": bool(successes[sample_idx]),
                "final_error_mean": float(final_error[sample_idx]),
                "num_high_plans": int(num_plans),
            }
        )
    for fig_idx, batch in enumerate(batched_indices(list(range(len(strip_images))), rows_per_figure)):
        out_path = out_dir / f"online_story_{fig_idx:02d}.png"
        draw_strip_grid(
            [strip_images[i] for i in batch],
            [strip_titles[i] for i in batch],
            [strip_labels[i] for i in batch],
            save_path=out_path,
            figure_title="Online Replanning",
        )
    return meta_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = args.output_dir / "teacher_vs_open_loop"
    generated_dir = args.output_dir / "generated_subgoal_acting"
    online_dir = args.output_dir / "online_hierarchical_logging"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    online_dir.mkdir(parents=True, exist_ok=True)

    teacher_data = np.load(args.teacher_artifact, allow_pickle=True)
    latent_dim = int(np.asarray(teacher_data["z_init"]).shape[-1])
    device = resolve_device(args.device)
    decoder, probe_ckpt = load_decoder(
        run_dir=args.probe_run_dir,
        probe_ckpt=args.probe_ckpt,
        latent_dim=latent_dim,
        device=device,
    )
    dataset = load_dataset(dataset_name=args.dataset_name, cache_dir=args.cache_dir)

    teacher_cases = render_teacher_stories(
        dataset=dataset,
        decoder=decoder,
        device=device,
        teacher_npz=args.teacher_artifact,
        out_dir=teacher_dir,
        goal_offset_steps=args.goal_offset_steps,
        frame_skip=args.frame_skip,
        num_cases=args.num_teacher_cases,
        rows_per_figure=args.rows_per_figure,
    )
    generated_cases = render_generated_stories(
        dataset=dataset,
        decoder=decoder,
        device=device,
        generated_npz=args.generated_artifact,
        out_dir=generated_dir,
        goal_offset_steps=args.goal_offset_steps,
        frame_skip=args.frame_skip,
        num_cases=args.num_generated_cases,
        rows_per_figure=args.rows_per_figure,
    )
    online_cases = render_online_stories(
        decoder=decoder,
        device=device,
        online_npz=args.online_artifact,
        out_dir=online_dir,
        num_cases=args.num_online_cases,
        rows_per_figure=args.rows_per_figure,
    )

    manifest = {
        "probe_run_dir": str(args.probe_run_dir),
        "probe_ckpt": str(probe_ckpt),
        "teacher_artifact": str(args.teacher_artifact),
        "generated_artifact": str(args.generated_artifact),
        "online_artifact": str(args.online_artifact) if args.online_artifact else None,
        "dataset_name": args.dataset_name,
        "goal_offset_steps": int(args.goal_offset_steps),
        "frame_skip": int(args.frame_skip),
        "rows_per_figure": int(args.rows_per_figure),
        "teacher_cases": teacher_cases,
        "generated_cases": generated_cases,
        "online_cases": online_cases,
        "online_trace_available": bool(online_cases),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
