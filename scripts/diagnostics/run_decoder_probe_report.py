from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics.decoder_probe_notebook_utils import (
    DEFAULT_ENV_DUMP,
    DEFAULT_HI_CKPT,
    DEFAULT_SHARED_CACHE,
    evaluate_bundle,
    export_diverse_waypoint_gallery,
    find_epoch_probe_checkpoints,
    load_environment_dump,
    load_probe_bundle,
    run_decoder_probe,
    sample_batch,
    save_checkpoint_comparison_figure,
    save_rollout_story_figure,
    save_waypoint_panel,
    summarize_environment_dump,
    summarize_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run decoder-probe analysis headlessly on CPU.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_SHARED_CACHE)
    parser.add_argument("--hi-ckpt", type=Path, default=Path(DEFAULT_HI_CKPT))
    parser.add_argument("--phase-a-run", type=Path, default=None)
    parser.add_argument("--phase-a-prefix", type=str, default="hi_decoder_probe_true_hope2_")
    parser.add_argument("--phase-a-probe-ckpt", type=Path, default=None)
    parser.add_argument("--phase-b-run", type=Path, default=None)
    parser.add_argument("--phase-b-probe-ckpt", type=Path, default=None)
    parser.add_argument("--phase-b-epoch", type=int, default=10)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_SHARED_CACHE / "runs")
    parser.add_argument("--phase-b-prefix", type=str, default="hi_decoder_probe_pred_exposed_hope2_")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-max-batches", type=int, default=12)
    parser.add_argument("--train-max-batches", type=int, default=6)
    parser.add_argument("--val-gallery-examples", type=int, default=12)
    parser.add_argument("--train-gallery-examples", type=int, default=6)
    parser.add_argument("--gallery-examples-per-batch", type=int, default=2)
    parser.add_argument("--comparison-samples", type=int, default=2)
    parser.add_argument("--epoch-compare", type=str, default="1,5,10")
    parser.add_argument("--env-dump", type=Path, default=DEFAULT_ENV_DUMP)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def latest_run_by_prefix(runs_root: Path, prefix: str) -> Path | None:
    candidates = sorted(
        [p for p in runs_root.glob(f"{prefix}*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def find_epoch_probe(run_dir: Path, epoch: int) -> Path | None:
    pattern = re.compile(r"_epoch_(\d+)_probe\.pt$")
    matches = []
    for ckpt in find_epoch_probe_checkpoints(run_dir):
        m = pattern.search(ckpt.name)
        if m and int(m.group(1)) == int(epoch):
            matches.append(ckpt)
    return sorted(matches)[-1] if matches else None


def checkpoint_epoch(ckpt: Path) -> int | None:
    m = re.search(r"_epoch_(\d+)_probe\.pt$", ckpt.name)
    return int(m.group(1)) if m else None


def parse_epoch_list(spec: str) -> list[int]:
    epochs = []
    for item in spec.split(','):
        item = item.strip()
        if item:
            epochs.append(int(item))
    return epochs


def save_series(series: pd.Series, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(series.to_dict(), indent=2, default=float))
    series.to_frame(name="value").to_csv(csv_path)


def load_bundle(run_dir: Path, probe_ckpt: Path | None, args: argparse.Namespace):
    return load_probe_bundle(
        run_dir=run_dir,
        probe_ckpt=probe_ckpt,
        hi_ckpt=args.hi_ckpt,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        device=args.device,
    )


def export_metrics(bundle, out_dir: Path, split: str, max_batches: int) -> dict:
    df = evaluate_bundle(bundle, split=split, max_batches=max_batches)
    df.to_csv(out_dir / f"{split}_metrics.csv", index=False)
    summary = summarize_metrics(df)
    save_series(summary, out_dir / f"{split}_summary.json", out_dir / f"{split}_summary.csv")
    return {
        "rows": len(df),
        "summary": summary,
        "metrics_csv": out_dir / f"{split}_metrics.csv",
        "summary_json": out_dir / f"{split}_summary.json",
    }


def export_batch_visuals(bundle, out_dir: Path, split: str, sample_count: int) -> dict:
    batch = sample_batch(bundle, split=split, batch_index=0)
    outputs = run_decoder_probe(bundle, batch)
    save_waypoint_panel(
        outputs,
        save_path=out_dir / f"{split}_waypoint_panel_rows2.png",
        num_rows=2,
        title=f"{split.title()} Decoder Probe Panel",
    )
    save_waypoint_panel(
        outputs,
        save_path=out_dir / f"{split}_waypoint_panel_rows6.png",
        num_rows=6,
        title=f"{split.title()} Decoder Probe Panel (6 rows)",
    )
    max_story_samples = min(sample_count, int(outputs["batch_size"]))
    for sample_index in range(max_story_samples):
        save_rollout_story_figure(
            outputs,
            save_path=out_dir / f"{split}_rollout_story_sample{sample_index}.png",
            sample_index=sample_index,
            title=f"{split.title()} Rollout Story Sample {sample_index}",
        )
    pd.Series(outputs["metrics"]).to_csv(out_dir / f"{split}_batch0_metrics.csv")
    return {"batch": batch, "outputs": outputs}


def export_gallery(bundle, out_dir: Path, split: str, num_examples: int, examples_per_batch: int) -> Path:
    gallery_dir = out_dir / f"{split}_gallery"
    gallery_df = export_diverse_waypoint_gallery(
        bundle,
        out_dir=gallery_dir,
        split=split,
        num_examples=num_examples,
        rows_per_panel=1,
        examples_per_batch=examples_per_batch,
        save_story_figures=True,
    )
    gallery_df.to_csv(gallery_dir / "gallery_index.csv", index=False)
    return gallery_dir


def export_epoch_comparison(
    run_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    batch: dict,
    label_prefix: str,
) -> list[Path]:
    requested_epochs = parse_epoch_list(args.epoch_compare)
    bundles = []
    for epoch in requested_epochs:
        ckpt = find_epoch_probe(run_dir, epoch)
        if ckpt is None:
            continue
        bundles.append((f"epoch {epoch}", load_bundle(run_dir, ckpt, args)))
    if len(bundles) < 2:
        return []
    paths = []
    for sample_index in range(args.comparison_samples):
        save_path = out_dir / f"{label_prefix}_epoch_compare_sample{sample_index}.png"
        save_checkpoint_comparison_figure(
            bundles,
            batch=batch,
            save_path=save_path,
            sample_index=sample_index,
            title=f"{label_prefix} Epoch Comparison",
        )
        paths.append(save_path)
    return paths


def write_report(report_path: Path, lines: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.hi_ckpt = args.hi_ckpt.expanduser().resolve()
    args.runs_root = args.runs_root.expanduser().resolve()
    args.phase_a_run = args.phase_a_run.expanduser().resolve() if args.phase_a_run else latest_run_by_prefix(args.runs_root, args.phase_a_prefix)
    if args.phase_a_run is None:
        raise FileNotFoundError(
            f"Could not locate a Phase A decoder probe run under {args.runs_root} "
            f"with prefix {args.phase_a_prefix!r}. Pass --phase-a-run explicitly."
        )

    phase_b_run = args.phase_b_run.expanduser().resolve() if args.phase_b_run else latest_run_by_prefix(args.runs_root, args.phase_b_prefix)
    phase_a_probe_ckpt = args.phase_a_probe_ckpt.expanduser().resolve() if args.phase_a_probe_ckpt else None
    phase_b_probe_ckpt = args.phase_b_probe_ckpt.expanduser().resolve() if args.phase_b_probe_ckpt else None
    if phase_b_run is not None and phase_b_probe_ckpt is None:
        phase_b_probe_ckpt = find_epoch_probe(phase_b_run, args.phase_b_epoch)

    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "output_dir": str(args.output_dir),
        "cache_dir": str(args.cache_dir),
        "hi_ckpt": str(args.hi_ckpt),
        "phase_a_run": str(args.phase_a_run),
        "phase_a_probe_ckpt": str(phase_a_probe_ckpt) if phase_a_probe_ckpt else None,
        "phase_b_run": str(phase_b_run) if phase_b_run else None,
        "phase_b_probe_ckpt": str(phase_b_probe_ckpt) if phase_b_probe_ckpt else None,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    env_dir = args.output_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    if args.env_dump.exists():
        env = load_environment_dump(args.env_dump)
        env_summary = summarize_environment_dump(env)
        save_series(env_summary, env_dir / "environment_summary.json", env_dir / "environment_summary.csv")
        (env_dir / "environment_dump.json").write_text(json.dumps(env, indent=2))

    phase_a = load_bundle(args.phase_a_run, phase_a_probe_ckpt, args)
    phase_a_dir = args.output_dir / "phase_a"
    phase_a_dir.mkdir(parents=True, exist_ok=True)
    phase_a_val = export_metrics(phase_a, phase_a_dir, "val", args.val_max_batches)
    phase_a_batch = export_batch_visuals(phase_a, phase_a_dir, "val", args.comparison_samples)
    phase_a_val_gallery = export_gallery(
        phase_a,
        phase_a_dir,
        "val",
        args.val_gallery_examples,
        args.gallery_examples_per_batch,
    )
    if not args.skip_train:
        export_metrics(phase_a, phase_a_dir, "train", args.train_max_batches)
        export_batch_visuals(phase_a, phase_a_dir, "train", args.comparison_samples)
        export_gallery(
            phase_a,
            phase_a_dir,
            "train",
            args.train_gallery_examples,
            args.gallery_examples_per_batch,
        )
    export_epoch_comparison(args.phase_a_run, phase_a_dir, args, phase_a_batch["batch"], "phase_a")

    phase_b = None
    phase_b_val = None
    if phase_b_run is not None and phase_b_probe_ckpt is not None:
        phase_b = load_bundle(phase_b_run, phase_b_probe_ckpt, args)
        phase_b_dir = args.output_dir / "phase_b"
        phase_b_dir.mkdir(parents=True, exist_ok=True)
        phase_b_val = export_metrics(phase_b, phase_b_dir, "val", args.val_max_batches)
        phase_b_batch = export_batch_visuals(phase_b, phase_b_dir, "val", args.comparison_samples)
        export_gallery(
            phase_b,
            phase_b_dir,
            "val",
            args.val_gallery_examples,
            args.gallery_examples_per_batch,
        )
        if not args.skip_train:
            export_metrics(phase_b, phase_b_dir, "train", args.train_max_batches)
            export_batch_visuals(phase_b, phase_b_dir, "train", args.comparison_samples)
            export_gallery(
                phase_b,
                phase_b_dir,
                "train",
                args.train_gallery_examples,
                args.gallery_examples_per_batch,
            )
        export_epoch_comparison(phase_b_run, phase_b_dir, args, phase_b_batch["batch"], "phase_b")

        compare_dir = args.output_dir / "comparisons"
        compare_dir.mkdir(parents=True, exist_ok=True)
        for sample_index in range(args.comparison_samples):
            save_checkpoint_comparison_figure(
                [("Phase A", phase_a), (f"Phase B epoch {args.phase_b_epoch}", phase_b)],
                batch=phase_a_batch["batch"],
                save_path=compare_dir / f"phase_a_vs_phase_b_sample{sample_index}.png",
                sample_index=sample_index,
                title="Phase A vs Phase B Decoded Predicted Waypoints",
            )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    lines = [
        "# Decoder Probe CPU Analysis",
        "",
        f"- Output directory: {args.output_dir}",
        f"- Shared/cache directory: {args.cache_dir}",
        f"- Frozen HOPE2 checkpoint: {args.hi_ckpt}",
        f"- Phase A run: {args.phase_a_run}",
        f"- Phase A probe checkpoint: {phase_a.probe_ckpt}",
        f"- Phase B run: {phase_b_run if phase_b_run else 'not used'}",
        f"- Phase B probe checkpoint: {phase_b_probe_ckpt if phase_b_probe_ckpt else 'not used'}",
        f"- Device: {args.device}",
        f"- Loader batch size: {args.batch_size}",
        f"- Loader workers: {args.num_workers}",
        "",
        "## Metrics",
        "",
        f"- Phase A val pixel_mse: {phase_a_val['summary'].get('pixel_mse', float('nan')):.6f}",
        f"- Phase A val psnr: {phase_a_val['summary'].get('psnr', float('nan')):.4f}",
    ]
    if phase_b_val is not None:
        lines.extend(
            [
                f"- Phase B val pixel_mse: {phase_b_val['summary'].get('pixel_mse', float('nan')):.6f}",
                f"- Phase B val psnr: {phase_b_val['summary'].get('psnr', float('nan')):.4f}",
                f"- Phase B val pixel_mse_pred: {phase_b_val['summary'].get('pixel_mse_pred', float('nan')):.6f}",
                f"- Phase B val psnr_pred: {phase_b_val['summary'].get('psnr_pred', float('nan')):.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Key Outputs",
            "",
            f"- Phase A directory: {args.output_dir / 'phase_a'}",
            f"- Phase A val gallery: {phase_a_val_gallery}",
            f"- Manifest: {manifest_path}",
        ]
    )
    if phase_b_run is not None and phase_b_probe_ckpt is not None:
        lines.extend(
            [
                f"- Phase B directory: {args.output_dir / 'phase_b'}",
                f"- Phase comparison directory: {args.output_dir / 'comparisons'}",
            ]
        )

    write_report(args.output_dir / 'report.md', lines)
    print(args.output_dir)


if __name__ == '__main__':
    main()
