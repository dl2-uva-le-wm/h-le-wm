from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


OFFLINE_CONFIGS = [
    {"kind": "teacher_vs_open_loop", "d": 25, "hh": 1, "lh": 2, "hns": 900, "hits": 20, "htopk": 10, "lns": 300, "lits": 30, "ltopk": 150},
    {"kind": "teacher_vs_open_loop", "d": 25, "hh": 2, "lh": 2, "hns": 900, "hits": 20, "htopk": 10, "lns": 300, "lits": 30, "ltopk": 150},
    {"kind": "teacher_vs_open_loop", "d": 50, "hh": 1, "lh": 2, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 30, "ltopk": 150},
    {"kind": "teacher_vs_open_loop", "d": 50, "hh": 2, "lh": 2, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 30, "ltopk": 150},
    {"kind": "dataset_subgoal_reachability", "d": 25, "hh": 1, "lh": 1, "hns": 900, "hits": 20, "htopk": 10, "lns": 300, "lits": 30, "ltopk": 150},
    {"kind": "dataset_subgoal_reachability", "d": 25, "hh": 1, "lh": 2, "hns": 900, "hits": 20, "htopk": 10, "lns": 300, "lits": 30, "ltopk": 150},
    {"kind": "dataset_subgoal_reachability", "d": 25, "hh": 1, "lh": 3, "hns": 900, "hits": 20, "htopk": 10, "lns": 300, "lits": 30, "ltopk": 150},
    {"kind": "dataset_subgoal_reachability", "d": 25, "hh": 1, "lh": 5, "hns": 900, "hits": 20, "htopk": 10, "lns": 300, "lits": 30, "ltopk": 150},
    {"kind": "dataset_subgoal_reachability", "d": 50, "hh": 1, "lh": 1, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 30, "ltopk": 150},
    {"kind": "dataset_subgoal_reachability", "d": 50, "hh": 1, "lh": 2, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 30, "ltopk": 150},
    {"kind": "dataset_subgoal_reachability", "d": 50, "hh": 1, "lh": 3, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 30, "ltopk": 150},
    {"kind": "dataset_subgoal_reachability", "d": 50, "hh": 1, "lh": 5, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 30, "ltopk": 150},
    {"kind": "generated_subgoal_reachability", "d": 25, "hh": 1, "lh": 2, "hns": 900, "hits": 20, "htopk": 10, "lns": 300, "lits": 30, "ltopk": 150},
    {"kind": "generated_subgoal_reachability", "d": 25, "hh": 2, "lh": 2, "hns": 900, "hits": 20, "htopk": 10, "lns": 300, "lits": 30, "ltopk": 150},
    {"kind": "generated_subgoal_reachability", "d": 50, "hh": 1, "lh": 2, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 30, "ltopk": 150},
    {"kind": "generated_subgoal_reachability", "d": 50, "hh": 2, "lh": 2, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 30, "ltopk": 150},
    {"kind": "macro_manifold", "d": 25, "hh": 2, "lh": 2, "hns": 900, "hits": 20, "htopk": 10, "lns": 300, "lits": 30, "ltopk": 150},
    {"kind": "macro_manifold", "d": 50, "hh": 2, "lh": 2, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 30, "ltopk": 150},
]

ACTING_CONFIGS = [
    {"kind": "oracle_subgoal_acting", "d": 50, "hh": 2, "lh": 2, "lrh": 1, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 20, "ltopk": 150},
    {"kind": "oracle_subgoal_acting", "d": 50, "hh": 2, "lh": 3, "lrh": 1, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 20, "ltopk": 150},
    {"kind": "oracle_subgoal_acting", "d": 50, "hh": 2, "lh": 5, "lrh": 1, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 20, "ltopk": 150},
    {"kind": "low_level_reality_gap", "d": 50, "hh": 2, "lh": 2, "lrh": 1, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 20, "ltopk": 150},
    {"kind": "low_level_reality_gap", "d": 50, "hh": 2, "lh": 3, "lrh": 1, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 20, "ltopk": 150},
    {"kind": "low_level_reality_gap", "d": 50, "hh": 2, "lh": 5, "lrh": 1, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 20, "ltopk": 150},
    {"kind": "generated_subgoal_acting", "d": 50, "hh": 1, "lh": 2, "lrh": 1, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 20, "ltopk": 150},
    {"kind": "generated_subgoal_acting", "d": 50, "hh": 2, "lh": 2, "lrh": 1, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 20, "ltopk": 150},
    {"kind": "online_hierarchical_logging", "d": 50, "hh": 1, "lh": 2, "lrh": 1, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 20, "ltopk": 150},
    {"kind": "online_hierarchical_logging", "d": 50, "hh": 2, "lh": 5, "lrh": 5, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 20, "ltopk": 150},
    {"kind": "online_hierarchical_logging", "d": 50, "hh": 2, "lh": 2, "lrh": 1, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 20, "ltopk": 150},
    {"kind": "online_hierarchical_logging", "d": 50, "hh": 2, "lh": 5, "lrh": 1, "hns": 1500, "hits": 40, "htopk": 10, "lns": 900, "lits": 20, "ltopk": 150},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run curated PushT diagnostics for the paper-ready surface.")
    parser.add_argument("mode", choices=["offline", "acting"])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def output_stem(cfg: dict[str, int | str]) -> str:
    stem = f"{cfg['kind']}_d{cfg['d']}_hh{cfg['hh']}_lh{cfg['lh']}"
    if "lrh" in cfg:
        stem += f"_lrh{cfg['lrh']}"
    return stem


def summary_name(kind: str) -> str:
    return f"summary_{kind}.tsv"


def run_command(argv: list[str]) -> None:
    pretty = " ".join(argv)
    print(pretty)
    proc = subprocess.run(argv, cwd=str(Path(__file__).resolve().parents[2]), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Diagnostic command failed with exit code {proc.returncode}")


def run_offline(args: argparse.Namespace) -> None:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    for cfg in OFFLINE_CONFIGS:
        stem = output_stem(cfg)
        argv = [
            sys.executable,
            "scripts/diagnostics/run_hi_diagnostic.py",
            "--policy", args.policy,
            "--experiment-kind", str(cfg["kind"]),
            "--dataset-name", "pusht_expert_train",
            "--goal-offset-steps", str(cfg["d"]),
            "--high-horizon", str(cfg["hh"]),
            "--low-horizon", str(cfg["lh"]),
            "--high-num-samples", str(cfg["hns"]),
            "--high-iters", str(cfg["hits"]),
            "--high-topk", str(cfg["htopk"]),
            "--low-num-samples", str(cfg["lns"]),
            "--low-iters", str(cfg["lits"]),
            "--low-topk", str(cfg["ltopk"]),
            "--seed", str(args.seed),
            "--device", args.device,
            "--save-json", str(output_root / f"{stem}.json"),
            "--save-npz", str(output_root / f"{stem}.npz"),
            "--append-tsv", str(output_root / summary_name(str(cfg["kind"]))),
        ]
        if args.cache_dir:
            argv.extend(["--cache-dir", str(args.cache_dir)])
        run_command(argv)


def run_acting(args: argparse.Namespace) -> None:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    for cfg in ACTING_CONFIGS:
        stem = output_stem(cfg)
        argv = [
            sys.executable,
            "scripts/diagnostics/run_hi_acting_diagnostic.py",
            "--policy", args.policy,
            "--experiment-kind", str(cfg["kind"]),
            "--dataset-name", "pusht_expert_train",
            "--eval-config", "h_le_wm/config/eval/hi_pusht.yaml",
            "--goal-offset-steps", str(cfg["d"]),
            "--eval-budget", "50",
            "--num-eval", "50",
            "--high-horizon", str(cfg["hh"]),
            "--low-horizon", str(cfg["lh"]),
            "--low-receding-horizon", str(cfg["lrh"]),
            "--high-num-samples", str(cfg["hns"]),
            "--high-iters", str(cfg["hits"]),
            "--high-topk", str(cfg["htopk"]),
            "--low-num-samples", str(cfg["lns"]),
            "--low-iters", str(cfg["lits"]),
            "--low-topk", str(cfg["ltopk"]),
            "--frame-skip", "5",
            "--seed", str(args.seed),
            "--device", args.device,
            "--save-json", str(output_root / f"{stem}.json"),
            "--save-npz", str(output_root / f"{stem}.npz"),
            "--append-tsv", str(output_root / summary_name(str(cfg["kind"]))),
        ]
        if args.cache_dir:
            argv.extend(["--cache-dir", str(args.cache_dir)])
        run_command(argv)


def main() -> int:
    args = build_parser().parse_args()
    if args.cache_dir:
        os.environ.setdefault("STABLEWM_HOME", str(args.cache_dir))
    if args.mode == "offline":
        run_offline(args)
        return 0
    run_acting(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
