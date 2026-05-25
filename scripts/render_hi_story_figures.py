#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
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

from decoder_probe_notebook_utils import find_latest_probe_checkpoint
from decoder_probe_notebook_utils import evaluate_bundle as evaluate_probe_bundle
from decoder_probe_notebook_utils import load_probe_bundle, run_decoder_probe, sample_batch
from h_le_wm.probe.model import LatentToPixelDecoder, denormalize_imagenet, load_decoder_state_dict
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
    p = argparse.ArgumentParser(description="Render paper-style HOPE2 diagnostic figures and tables.")
    p.add_argument("--probe-run-dir", type=Path, required=True)
    p.add_argument("--probe-ckpt", type=Path, default=None)
    p.add_argument("--teacher-artifact", type=Path, required=True)
    p.add_argument("--oracle-artifact", type=Path, required=True)
    p.add_argument("--generated-artifact", type=Path, required=True)
    p.add_argument("--online-artifact", type=Path, required=True)
    p.add_argument("--paper-tables-dir", type=Path, required=True)
    p.add_argument("--baseline-md", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dataset-name", default="pusht_expert_train")
    p.add_argument("--goal-offset-steps", type=int, default=50)
    p.add_argument("--frame-skip", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--probe-split", default="val", choices=["train", "val"])
    p.add_argument("--probe-batch-index", type=int, default=0)
    p.add_argument("--probe-metric-batches", type=int, default=8)
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
    decoder = LatentToPixelDecoder(latent_dim=int(latent_dim), img_size=int(cfg.img_size), **decoder_cfg)
    decoder.load_state_dict(load_decoder_state_dict(ckpt), strict=True)
    return decoder.to(device).eval(), ckpt


@torch.inference_mode()
def decode_latents(decoder: torch.nn.Module, latents: np.ndarray, device: torch.device, *, batch_size: int = 256) -> np.ndarray:
    arr = np.asarray(latents, dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[-1])
    outs = []
    for start in range(0, flat.shape[0], batch_size):
        chunk = torch.from_numpy(flat[start : start + batch_size]).to(device)
        decoded = decoder(chunk)
        vis = denormalize_imagenet(decoded).detach().cpu().permute(0, 2, 3, 1).numpy()
        outs.append(np.clip(vis, 0.0, 1.0))
    return np.concatenate(outs, axis=0).reshape(*arr.shape[:-1], *outs[0].shape[1:])


def fetch_pixels(dataset, row_indices: np.ndarray) -> np.ndarray:
    rows = get_row_data_safe(dataset, np.asarray(row_indices, dtype=np.int64))
    return np.stack([normalize_image(px) for px in np.asarray(rows["pixels"])], axis=0)


def hide_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_grid_figure(
    *,
    rows: list[list[np.ndarray | None]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    out_path: Path,
    tile_w: float = 2.2,
    tile_h: float = 2.35,
) -> None:
    nrows = len(rows)
    ncols = len(col_labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(tile_w * ncols, tile_h * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontsize=10, fontweight="bold")
    for r in range(nrows):
        axes[r, 0].set_ylabel(row_labels[r], fontsize=10, fontweight="bold")
        for c in range(ncols):
            image = rows[r][c] if c < len(rows[r]) else None
            if image is not None:
                axes[r, c].imshow(image)
            hide_axes(axes[r, c])
    fig.suptitle(title, fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def baseline_d50_best(path: Path) -> float | None:
    if path.suffix.lower() == ".csv":
        with path.open() as f:
            rows = list(csv.DictReader(f))
        values = []
        for row in rows:
            if str(row.get("goal_offset_steps", "")).strip() != "50":
                continue
            try:
                values.append(float(row.get("success_rate_mean", "")))
            except ValueError:
                continue
        return max(values) if values else None

    text = path.read_text()
    m = re.search(r"\| `D50` \| 6 \| `([0-9.]+)` \| `([0-9.]+)` \| `([0-9.]+)` \|", text)
    return float(m.group(2)) if m else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def filter_hope2(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        policy = row.get("policy", "")
        if "hi_lewm_p2_train_hope2_22253175" in policy:
            out.append(row)
    return out


def choose_top_indices(scores: np.ndarray, n: int) -> list[int]:
    order = np.argsort(scores)[::-1]
    return [int(x) for x in order[: min(n, len(order))]]


def select_offline_cases(teacher_err: np.ndarray, open_true_err: np.ndarray, open_cem_err: np.ndarray, n: int) -> list[int]:
    score = (open_cem_err - teacher_err) + 0.35 * np.maximum(open_true_err - teacher_err, 0.0)
    return choose_top_indices(score, n)


def select_generated_cases(oracle_success: np.ndarray, generated_success: np.ndarray, offset_error: np.ndarray, final_error: np.ndarray, n: int) -> list[int]:
    buckets: list[int] = []
    buckets.extend(np.where(oracle_success & ~generated_success)[0].tolist())
    buckets.extend(np.where(oracle_success & generated_success)[0].tolist())
    buckets.extend(np.where(~oracle_success & ~generated_success)[0].tolist())
    seen: list[int] = []
    for idx in buckets:
        if idx not in seen:
            seen.append(int(idx))
    seen = sorted(seen, key=lambda i: (not (oracle_success[i] and not generated_success[i]), -abs(offset_error[i]), -final_error[i]))
    return seen[: min(n, len(seen))]


def select_online_cases(success: np.ndarray, final_error: np.ndarray, n: int) -> list[int]:
    score = (~success).astype(np.float32) * 10.0 + final_error
    return choose_top_indices(score, n)


def stage_target_rows(start_row: int, horizon: int, goal_offset_steps: int, frame_skip: int) -> np.ndarray:
    spans = partition_total(max(1, int(math.ceil(goal_offset_steps / frame_skip))), horizon)
    offsets = np.cumsum([span * frame_skip for span in spans]).astype(np.int64)
    return start_row + offsets


def nearest_offsets(generated_stage_targets: np.ndarray, future_latents: np.ndarray) -> np.ndarray:
    generated = torch.from_numpy(np.asarray(generated_stage_targets, dtype=np.float32))
    future = torch.from_numpy(np.asarray(future_latents[:, 1:, :], dtype=np.float32))
    out = []
    for stage_idx in range(generated.shape[1]):
        dist = torch.cdist(generated[:, stage_idx : stage_idx + 1], future).squeeze(1)
        out.append((torch.argmin(dist, dim=1) + 1).detach().cpu().numpy())
    return np.stack(out, axis=1)


def render_offline_forecast(
    *,
    dataset,
    decoder,
    device: torch.device,
    teacher_npz: Path,
    out_dir: Path,
    goal_offset_steps: int,
    frame_skip: int,
    rows: int,
) -> dict[str, object]:
    data = np.load(teacher_npz, allow_pickle=True)
    starts = np.asarray(data["starts"], dtype=np.int64)
    teacher_err = np.asarray(data["teacher_err"], dtype=np.float32)
    open_true_err = np.asarray(data["open_true_err"], dtype=np.float32)
    open_cem_err = np.asarray(data["open_cem_err"], dtype=np.float32)
    teacher_pred = np.asarray(data["teacher_pred"], dtype=np.float32)
    open_true_pred = np.asarray(data["open_true_pred"], dtype=np.float32)
    open_cem_pred = np.asarray(data["open_cem_pred"], dtype=np.float32)
    horizon = int(teacher_pred.shape[1])
    chosen = select_offline_cases(
        teacher_err.mean(axis=1),
        open_true_err.mean(axis=1),
        open_cem_err.mean(axis=1),
        rows,
    )
    teacher_dec = decode_latents(decoder, teacher_pred[chosen], device)
    open_true_dec = decode_latents(decoder, open_true_pred[chosen], device)
    open_cem_dec = decode_latents(decoder, open_cem_pred[chosen], device)

    outputs = {}
    for stage_idx in range(horizon):
        grid_rows = []
        row_labels = []
        for local_idx, sample_idx in enumerate(chosen):
            target_row = stage_target_rows(int(starts[sample_idx]), horizon, goal_offset_steps, frame_skip)[stage_idx]
            context_img = fetch_pixels(dataset, np.array([starts[sample_idx]]))[0]
            target_img = fetch_pixels(dataset, np.array([target_row]))[0]
            grid_rows.append(
                [
                    context_img,
                    target_img,
                    teacher_dec[local_idx, stage_idx],
                    open_true_dec[local_idx, stage_idx],
                    open_cem_dec[local_idx, stage_idx],
                ]
            )
            row_labels.append(
                f"Episode {local_idx + 1}\nTF {teacher_err[sample_idx, stage_idx]:.03f}\nCEM {open_cem_err[sample_idx, stage_idx]:.03f}"
            )
        filename = f"offline_forecast_stage_{stage_idx + 1}.png"
        save_grid_figure(
            rows=grid_rows,
            row_labels=row_labels,
            col_labels=["Context", "Target", "Teacher", "Open-Loop True", "Open-Loop CEM"],
            title=f"Offline Forecast, Stage {stage_idx + 1}",
            out_path=out_dir / filename,
        )
        outputs[f"stage_{stage_idx + 1}"] = {
            "file": filename,
            "sample_indices": chosen,
        }
    return outputs


def render_probe_sanity(
    *,
    probe_run_dir: Path,
    probe_ckpt: Path | None,
    device: torch.device,
    cache_dir: str | None,
    split: str,
    batch_index: int,
    rows: int,
    metric_batches: int,
    out_dir: Path,
    tables_dir: Path,
) -> dict[str, object]:
    bundle = load_probe_bundle(
        run_dir=probe_run_dir,
        probe_ckpt=probe_ckpt,
        cache_dir=cache_dir,
        batch_size=128,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        device=device,
    )
    batch = sample_batch(bundle, split=split, batch_index=batch_index)
    outputs = run_decoder_probe(bundle, batch)
    nrows = min(rows, int(outputs["target_vis"].shape[0]))
    grid_rows = []
    row_labels = []
    for row_idx in range(nrows):
        context = outputs["context_vis"][row_idx].permute(1, 2, 0).numpy()
        target = outputs["target_vis"][row_idx].permute(1, 2, 0).numpy()
        dec_true = outputs["decoded_true_vis"][row_idx].permute(1, 2, 0).numpy()
        dec_pred = outputs["decoded_pred_vis"][row_idx].permute(1, 2, 0).numpy()
        err = outputs["error_map"][row_idx].numpy()
        err_rgb = plt.get_cmap("magma")(np.clip(err, 0.0, np.percentile(err, 99)))[:, :, :3]
        grid_rows.append([context, target, dec_true, dec_pred, err_rgb])
        row_labels.append(f"Example {row_idx + 1}")
    filename = "probe_sanity.png"
    save_grid_figure(
        rows=grid_rows,
        row_labels=row_labels,
        col_labels=["Context", "True", "Decoded True", "Decoded Pred", "Abs Error"],
        title="Decoder Probe Sanity Check",
        out_path=out_dir / filename,
    )

    metric_df = evaluate_probe_bundle(bundle, split=split, max_batches=metric_batches)
    metrics_path = tables_dir / "decoder_probe_metrics.csv"
    metric_df.to_csv(metrics_path, index=False)
    summary_path = tables_dir / "decoder_probe_summary.csv"
    summary_rows = [
        ("pixel_mse_true_mean", float(metric_df["pixel_mse_true"].mean())),
        ("pixel_mse_pred_mean", float(metric_df["pixel_mse_pred"].mean())),
        ("psnr_true_mean", float(metric_df["psnr_true"].mean())),
        ("psnr_pred_mean", float(metric_df["psnr_pred"].mean())),
        ("latent_gap_mean", float(metric_df["latent_gap"].mean())),
    ]
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(summary_rows)
    return {
        "file": filename,
        "metrics_file": metrics_path.name,
        "summary_file": summary_path.name,
        "split": split,
        "batch_index": int(batch_index),
    }


def render_oracle_vs_generated(
    *,
    dataset,
    decoder,
    device: torch.device,
    oracle_npz: Path,
    generated_npz: Path,
    goal_offset_steps: int,
    frame_skip: int,
    rows: int,
    out_dir: Path,
) -> dict[str, object]:
    oracle = np.load(oracle_npz, allow_pickle=True)
    generated = np.load(generated_npz, allow_pickle=True)
    sampled_indices = np.asarray(generated["sampled_indices"], dtype=np.int64)
    oracle_success = np.asarray(oracle["episode_successes"], dtype=bool)
    generated_success = np.asarray(generated["episode_successes"], dtype=bool)
    future_latents = np.asarray(generated["future_latents"], dtype=np.float32)
    generated_targets = np.asarray(generated["generated_stage_targets"], dtype=np.float32)
    stage_end_actual = np.asarray(generated["stage_end_actual_latents"], dtype=np.float32)
    oracle_targets = np.asarray(oracle["oracle_stage_targets"], dtype=np.float32)
    oracle_final = np.asarray(oracle["final_latent"], dtype=np.float32)
    generated_final = np.asarray(generated["final_latent"], dtype=np.float32)
    goal_latent = np.asarray(generated["goal_latent"], dtype=np.float32)
    nearest = nearest_offsets(generated_targets, future_latents)
    expected = stage_target_rows(0, int(generated_targets.shape[1]), goal_offset_steps, frame_skip) - 0
    offset_error = nearest[:, 0] - expected[0]
    final_error = np.mean((generated_final - goal_latent) ** 2, axis=1)
    chosen = select_generated_cases(oracle_success, generated_success, offset_error, final_error, rows)

    oracle_stage1_dec = decode_latents(decoder, oracle_targets[chosen, 0, :], device)
    oracle_final_dec = decode_latents(decoder, oracle_final[chosen], device)
    generated_stage1_dec = decode_latents(decoder, generated_targets[chosen, 0, :], device)
    generated_reached_dec = decode_latents(decoder, stage_end_actual[0, chosen, :], device)
    generated_final_dec = decode_latents(decoder, generated_final[chosen], device)

    grid_rows = []
    row_labels = []
    for local_idx, sample_idx in enumerate(chosen):
        start_row = int(sampled_indices[sample_idx])
        target_row = start_row + int(expected[0])
        context_img = fetch_pixels(dataset, np.array([start_row]))[0]
        target_img = fetch_pixels(dataset, np.array([target_row]))[0]
        grid_rows.append(
            [
                context_img,
                target_img,
                oracle_stage1_dec[local_idx],
                oracle_final_dec[local_idx],
                generated_stage1_dec[local_idx],
                generated_reached_dec[local_idx],
                generated_final_dec[local_idx],
            ]
        )
        row_labels.append(
            f"Episode {local_idx + 1}\nOracle {'ok' if oracle_success[sample_idx] else 'fail'}\nGen {'ok' if generated_success[sample_idx] else 'fail'}"
        )
    filename = "oracle_vs_generated.png"
    save_grid_figure(
        rows=grid_rows,
        row_labels=row_labels,
        col_labels=["Context", "Target", "Oracle", "Oracle Final", "Generated", "Generated Reach", "Generated Final"],
        title="Oracle Subgoal vs Generated Subgoal Acting",
        out_path=out_dir / filename,
    )
    return {"file": filename, "sample_indices": chosen}


def render_temporal_validity(
    *,
    dataset,
    decoder,
    device: torch.device,
    generated_npz: Path,
    goal_offset_steps: int,
    frame_skip: int,
    rows: int,
    out_dir: Path,
) -> dict[str, object]:
    generated = np.load(generated_npz, allow_pickle=True)
    sampled_indices = np.asarray(generated["sampled_indices"], dtype=np.int64)
    future_latents = np.asarray(generated["future_latents"], dtype=np.float32)
    generated_targets = np.asarray(generated["generated_stage_targets"], dtype=np.float32)
    stage_end_actual = np.asarray(generated["stage_end_actual_latents"], dtype=np.float32)
    final_latent = np.asarray(generated["final_latent"], dtype=np.float32)
    successes = np.asarray(generated["episode_successes"], dtype=bool)
    nearest = nearest_offsets(generated_targets, future_latents)
    expected = stage_target_rows(0, int(generated_targets.shape[1]), goal_offset_steps, frame_skip) - 0
    offset_error = nearest[:, 0] - expected[0]
    final_error = np.mean((final_latent - np.asarray(generated["goal_latent"], dtype=np.float32)) ** 2, axis=1)
    chosen = choose_top_indices((~successes).astype(np.float32) * 10.0 + np.abs(offset_error) + 0.25 * final_error, rows)

    generated_dec = decode_latents(decoder, generated_targets[chosen, 0, :], device)
    nearest_dec = decode_latents(decoder, future_latents[chosen, nearest[chosen, 0], :], device)
    reached_dec = decode_latents(decoder, stage_end_actual[0, chosen, :], device)
    final_dec = decode_latents(decoder, final_latent[chosen], device)

    grid_rows = []
    row_labels = []
    for local_idx, sample_idx in enumerate(chosen):
        start_row = int(sampled_indices[sample_idx])
        target_row = start_row + int(expected[0])
        context_img = fetch_pixels(dataset, np.array([start_row]))[0]
        target_img = fetch_pixels(dataset, np.array([target_row]))[0]
        grid_rows.append(
            [
                context_img,
                target_img,
                generated_dec[local_idx],
                nearest_dec[local_idx],
                reached_dec[local_idx],
                final_dec[local_idx],
            ]
        )
        row_labels.append(f"Episode {local_idx + 1}\nOffset {int(offset_error[sample_idx]):+d}\n{'fail' if not successes[sample_idx] else 'ok'}")
    filename = "temporal_validity.png"
    save_grid_figure(
        rows=grid_rows,
        row_labels=row_labels,
        col_labels=["Context", "Target", "Generated", "Closest Match", "Reached", "Final Goal"],
        title="Generated Subgoal Temporal Validity",
        out_path=out_dir / filename,
    )
    return {"file": filename, "sample_indices": chosen}


def render_online_replanning(
    *,
    dataset,
    decoder,
    device: torch.device,
    online_npz: Path,
    rows: int,
    out_dir: Path,
) -> dict[str, object]:
    online = np.load(online_npz, allow_pickle=True)
    sampled_indices = np.asarray(online["sampled_indices"], dtype=np.int64)
    success = np.asarray(online["episode_successes"], dtype=bool)
    final_latent = np.asarray(online["final_latent"], dtype=np.float32)
    goal_latent = np.asarray(online["goal_latent"], dtype=np.float32)
    high_plan_current = np.asarray(online["high_plan_current_latents"], dtype=np.float32)
    high_plan_subgoal = np.asarray(online["high_plan_subgoal_latents"], dtype=np.float32)
    low_block_actual = np.asarray(online["low_block_actual_latents"], dtype=np.float32)
    low_block_high_plan_ids = np.asarray(online["low_block_high_plan_ids"], dtype=np.int64)
    chosen = select_online_cases(success, np.mean((final_latent - goal_latent) ** 2, axis=1), rows)

    goal_dec = decode_latents(decoder, goal_latent[chosen], device)
    subgoal_dec = decode_latents(decoder, np.transpose(high_plan_subgoal[:2, chosen, :], (1, 0, 2)), device)
    reached_dec = decode_latents(decoder, np.transpose(low_block_actual[:2, chosen, :], (1, 0, 2)), device)

    grid_rows = []
    row_labels = []
    for local_idx, sample_idx in enumerate(chosen):
        context_img = fetch_pixels(dataset, np.array([int(sampled_indices[sample_idx])]))[0]
        row = [context_img, goal_dec[local_idx]]
        titles_match = []
        for plan_idx in range(2):
            row.append(subgoal_dec[local_idx, plan_idx])
            reached = None
            match = np.where(low_block_high_plan_ids == plan_idx)[0]
            if match.size > 0 and match[0] < reached_dec.shape[1]:
                reached = reached_dec[local_idx, match[0]]
            row.append(reached)
            titles_match.append(plan_idx)
        grid_rows.append(row)
        row_labels.append(f"Episode {local_idx + 1}\n{'fail' if not success[sample_idx] else 'ok'}")
    filename = "online_replanning.png"
    save_grid_figure(
        rows=grid_rows,
        row_labels=row_labels,
        col_labels=["Context", "Goal", "Plan 1", "Reached 1", "Plan 2", "Reached 2"],
        title="Online Replanning Instability",
        out_path=out_dir / filename,
    )
    return {"file": filename, "sample_indices": chosen}


def write_story_tables(
    *,
    paper_tables_dir: Path,
    teacher_npz: Path,
    baseline_md: Path,
    output_dir: Path,
) -> dict[str, str]:
    out = {}
    teacher_rows = filter_hope2(read_csv_rows(paper_tables_dir / "teacher_vs_open_loop.csv"))
    oracle_rows = filter_hope2(read_csv_rows(paper_tables_dir / "oracle_subgoal_acting.csv"))
    generated_rows = filter_hope2(read_csv_rows(paper_tables_dir / "generated_subgoal_acting.csv"))
    online_rows = filter_hope2(read_csv_rows(paper_tables_dir / "online_hierarchical_logging.csv"))

    def pick(rows: list[dict[str, str]], **conds: str) -> dict[str, str]:
        for row in rows:
            if all(str(row.get(k, "")) == str(v) for k, v in conds.items()):
                return row
        raise KeyError(f"No row matching {conds}")

    failure_path = output_dir / "tables" / "failure_decomposition.csv"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "success_rate", "final_or_terminal_error", "reality_gap", "subgoal_churn"])
        baseline = baseline_d50_best(baseline_md)
        oracle = pick(oracle_rows, goal_offset_steps="50", high_horizon="2", low_horizon="2", low_receding_horizon="1")
        generated = pick(generated_rows, goal_offset_steps="50", high_horizon="2", low_horizon="2", low_receding_horizon="1")
        online = pick(online_rows, goal_offset_steps="50", high_horizon="2", low_horizon="2", low_receding_horizon="1")
        w.writerow(["baseline_best_d50", baseline, "", "", ""])
        w.writerow(["oracle_subgoal_acting", oracle["success_rate"], oracle["final_terminal_latent_error_mean"], oracle["reality_gap_mean"], ""])
        w.writerow(["generated_subgoal_acting", generated["success_rate"], generated["final_terminal_latent_error_mean"], "", ""])
        w.writerow(["online_hierarchical", online["success_rate"], online["final_terminal_latent_error_mean"], online["mean_reality_gap"], online["mean_subgoal_churn_mse"]])
    out["failure_decomposition"] = str(failure_path.name)

    forecast_path = output_dir / "tables" / "high_level_forecast.csv"
    with forecast_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "teacher_mse", "open_true_mse", "open_cem_mse"])
        for row in teacher_rows:
            key = f"D{row['goal_offset_steps']}_H{row['high_horizon']}"
            w.writerow([key, row["teacher_forced_mse_mean"], row["open_loop_true_mse_mean"], row["open_loop_cem_mse_mean"]])
    out["high_level_forecast"] = str(forecast_path.name)

    validity_path = output_dir / "tables" / "generated_subgoal_validity.csv"
    with validity_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "step1_offset_error", "step2_offset_error", "success_rate"])
        for row in generated_rows:
            key = f"D{row['goal_offset_steps']}_H{row['high_horizon']}"
            w.writerow([key, row.get("step1_offset_error_token_mean", ""), row.get("step2_offset_error_token_mean", ""), row["success_rate"]])
    out["generated_subgoal_validity"] = str(validity_path.name)

    online_path = output_dir / "tables" / "online_replanning.csv"
    with online_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "success_rate", "subgoal_churn_mse", "reality_gap"])
        for row in online_rows:
            key = f"D{row['goal_offset_steps']}_H{row['high_horizon']}_L{row['low_horizon']}_R{row['low_receding_horizon']}"
            w.writerow([key, row["success_rate"], row["mean_subgoal_churn_mse"], row["mean_reality_gap"]])
    out["online_replanning"] = str(online_path.name)

    teacher_np = np.load(teacher_npz, allow_pickle=True)
    teacher_err = np.asarray(teacher_np["teacher_err"], dtype=np.float32)
    open_true_err = np.asarray(teacher_np["open_true_err"], dtype=np.float32)
    open_cem_err = np.asarray(teacher_np["open_cem_err"], dtype=np.float32)
    compounding_path = output_dir / "tables" / "open_loop_cem_compounding.csv"
    with compounding_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "stage",
                "teacher_mse_mean",
                "open_true_mse_mean",
                "open_cem_mse_mean",
                "delta_open_true_vs_teacher",
                "delta_open_cem_vs_teacher",
                "delta_open_cem_vs_open_true",
            ]
        )
        for stage_idx in range(teacher_err.shape[1]):
            tf = float(teacher_err[:, stage_idx].mean())
            ot = float(open_true_err[:, stage_idx].mean())
            cem = float(open_cem_err[:, stage_idx].mean())
            w.writerow(
                [
                    f"stage_{stage_idx + 1}",
                    tf,
                    ot,
                    cem,
                    ot - tf,
                    cem - tf,
                    cem - ot,
                ]
            )
        tf_all = float(teacher_err.mean())
        ot_all = float(open_true_err.mean())
        cem_all = float(open_cem_err.mean())
        w.writerow(
            [
                "overall",
                tf_all,
                ot_all,
                cem_all,
                ot_all - tf_all,
                cem_all - tf_all,
                cem_all - ot_all,
            ]
        )
    out["open_loop_cem_compounding"] = str(compounding_path.name)

    dynamics_path = output_dir / "tables" / "open_loop_error_dynamics.csv"
    with dynamics_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "stage1_mse_mean",
                "stage2_mse_mean",
                "overall_mse_mean",
                "stage2_minus_stage1",
                "stage2_over_stage1",
                "pct_episodes_stage2_gt_stage1",
            ]
        )
        for method_name, arr in [
            ("teacher", teacher_err),
            ("open_loop_true", open_true_err),
            ("open_loop_cem", open_cem_err),
        ]:
            stage1 = arr[:, 0]
            stage2 = arr[:, 1]
            stage1_mean = float(stage1.mean())
            stage2_mean = float(stage2.mean())
            w.writerow(
                [
                    method_name,
                    stage1_mean,
                    stage2_mean,
                    float(arr.mean()),
                    float((stage2 - stage1).mean()),
                    float(stage2_mean / max(stage1_mean, 1e-12)),
                    float((stage2 > stage1).mean()),
                ]
            )
    out["open_loop_error_dynamics"] = str(dynamics_path.name)

    comparison_path = output_dir / "tables" / "open_loop_method_comparison.csv"
    with comparison_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "comparison",
                "stage1_delta_mean",
                "stage1_pct_worse",
                "stage2_delta_mean",
                "stage2_pct_worse",
                "overall_delta_mean",
                "overall_pct_worse",
            ]
        )
        for label, lhs, rhs in [
            ("open_true_minus_teacher", open_true_err, teacher_err),
            ("open_cem_minus_teacher", open_cem_err, teacher_err),
            ("open_cem_minus_open_true", open_cem_err, open_true_err),
        ]:
            stage1_diff = lhs[:, 0] - rhs[:, 0]
            stage2_diff = lhs[:, 1] - rhs[:, 1]
            overall_diff = lhs.mean(axis=1) - rhs.mean(axis=1)
            w.writerow(
                [
                    label,
                    float(stage1_diff.mean()),
                    float((stage1_diff > 0).mean()),
                    float(stage2_diff.mean()),
                    float((stage2_diff > 0).mean()),
                    float(overall_diff.mean()),
                    float((overall_diff > 0).mean()),
                ]
            )
    out["open_loop_method_comparison"] = str(comparison_path.name)

    markdown_path = output_dir / "tables" / "story_tables.md"
    with markdown_path.open("w") as f:
        f.write("# Story Tables\n\n")
        f.write(f"- Failure decomposition: `{failure_path.name}`\n")
        f.write(f"- High-level forecast: `{forecast_path.name}`\n")
        f.write(f"- Open-loop CEM compounding: `{compounding_path.name}`\n")
        f.write(f"- Open-loop error dynamics: `{dynamics_path.name}`\n")
        f.write(f"- Open-loop method comparison: `{comparison_path.name}`\n")
        f.write(f"- Generated subgoal validity: `{validity_path.name}`\n")
        f.write(f"- Online replanning: `{online_path.name}`\n")
    out["story_tables_md"] = str(markdown_path.name)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = fig_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = fig_dir / "offline"
    acting_dir = fig_dir / "acting"
    online_dir = fig_dir / "online"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    acting_dir.mkdir(parents=True, exist_ok=True)
    online_dir.mkdir(parents=True, exist_ok=True)

    teacher_data = np.load(args.teacher_artifact, allow_pickle=True)
    latent_dim = int(np.asarray(teacher_data["z_init"]).shape[-1])
    device = resolve_device(args.device)
    decoder, probe_ckpt = load_decoder(run_dir=args.probe_run_dir, probe_ckpt=args.probe_ckpt, latent_dim=latent_dim, device=device)
    dataset = load_dataset(dataset_name=args.dataset_name, cache_dir=args.cache_dir)

    probe_info = render_probe_sanity(
        probe_run_dir=args.probe_run_dir,
        probe_ckpt=args.probe_ckpt,
        device=device,
        cache_dir=args.cache_dir,
        split=args.probe_split,
        batch_index=args.probe_batch_index,
        rows=args.rows,
        metric_batches=args.probe_metric_batches,
        out_dir=probe_dir,
        tables_dir=tables_dir,
    )

    offline_info = render_offline_forecast(
        dataset=dataset,
        decoder=decoder,
        device=device,
        teacher_npz=args.teacher_artifact,
        out_dir=teacher_dir,
        goal_offset_steps=args.goal_offset_steps,
        frame_skip=args.frame_skip,
        rows=args.rows,
    )
    oracle_vs_generated_info = render_oracle_vs_generated(
        dataset=dataset,
        decoder=decoder,
        device=device,
        oracle_npz=args.oracle_artifact,
        generated_npz=args.generated_artifact,
        goal_offset_steps=args.goal_offset_steps,
        frame_skip=args.frame_skip,
        rows=args.rows,
        out_dir=acting_dir,
    )
    temporal_info = render_temporal_validity(
        dataset=dataset,
        decoder=decoder,
        device=device,
        generated_npz=args.generated_artifact,
        goal_offset_steps=args.goal_offset_steps,
        frame_skip=args.frame_skip,
        rows=args.rows,
        out_dir=acting_dir,
    )
    online_info = render_online_replanning(
        dataset=dataset,
        decoder=decoder,
        device=device,
        online_npz=args.online_artifact,
        rows=args.rows,
        out_dir=online_dir,
    )
    table_info = write_story_tables(
        paper_tables_dir=args.paper_tables_dir,
        teacher_npz=args.teacher_artifact,
        baseline_md=args.baseline_md,
        output_dir=args.output_dir,
    )

    manifest = {
        "probe_run_dir": str(args.probe_run_dir),
        "probe_ckpt": str(probe_ckpt),
        "teacher_artifact": str(args.teacher_artifact),
        "oracle_artifact": str(args.oracle_artifact),
        "generated_artifact": str(args.generated_artifact),
        "online_artifact": str(args.online_artifact),
        "paper_tables_dir": str(args.paper_tables_dir),
        "rows": int(args.rows),
        "figures": {
            "probe": probe_info,
            "offline": offline_info,
            "oracle_vs_generated": oracle_vs_generated_info,
            "temporal_validity": temporal_info,
            "online": online_info,
        },
        "tables": table_info,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
