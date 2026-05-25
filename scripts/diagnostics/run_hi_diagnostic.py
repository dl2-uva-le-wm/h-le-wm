#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from hi_diagnostics import DEFAULT_SUBGOAL_OFFSETS, DiagnosticConfig, run_diagnostic


def parse_subgoal_offsets(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return DEFAULT_SUBGOAL_OFFSETS
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        return DEFAULT_SUBGOAL_OFFSETS
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified offline diagnostics for hierarchical PushT checkpoints.")
    p.add_argument("--policy", required=True, help="Policy path for AutoCostModel (relative to STABLEWM_HOME).")
    p.add_argument(
        "--experiment-kind",
        required=True,
        choices=[
            "macro_manifold",
            "teacher_vs_open_loop",
            "dataset_subgoal_reachability",
            "generated_subgoal_reachability",
        ],
    )
    p.add_argument("--dataset-name", default="pusht_expert_train")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--goal-offset-steps", type=int, default=25)
    p.add_argument("--high-horizon", type=int, default=1)
    p.add_argument("--low-horizon", type=int, default=2)
    p.add_argument("--high-num-samples", type=int, default=900)
    p.add_argument("--high-iters", type=int, default=20)
    p.add_argument("--high-topk", type=int, default=10)
    p.add_argument("--low-num-samples", type=int, default=300)
    p.add_argument("--low-iters", type=int, default=30)
    p.add_argument("--low-topk", type=int, default=150)
    p.add_argument("--num-eval-samples", type=int, default=256)
    p.add_argument("--num-empirical-chunks", type=int, default=4096)
    p.add_argument("--reference-latent-pool-size", type=int, default=4096)
    p.add_argument("--cem-elite-frac", type=float, default=0.1)
    p.add_argument(
        "--cem-bound-mode",
        default="none",
        choices=["none", "q01_q99", "q05_q95"],
    )
    p.add_argument("--frame-skip", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--subgoal-offsets", default="2,3,5")
    p.add_argument("--max-cov-dim-for-json", type=int, default=64)
    p.add_argument("--save-json", default=None)
    p.add_argument("--save-npz", default=None)
    p.add_argument("--append-tsv", default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = DiagnosticConfig(
        policy=args.policy,
        experiment_kind=args.experiment_kind,
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        img_size=args.img_size,
        num_eval_samples=args.num_eval_samples,
        num_empirical_chunks=args.num_empirical_chunks,
        goal_offset_steps=args.goal_offset_steps,
        high_horizon=args.high_horizon,
        low_horizon=args.low_horizon,
        high_num_samples=args.high_num_samples,
        high_iters=args.high_iters,
        high_topk=args.high_topk,
        low_num_samples=args.low_num_samples,
        low_iters=args.low_iters,
        low_topk=args.low_topk,
        cem_elite_frac=args.cem_elite_frac,
        cem_bound_mode=args.cem_bound_mode,
        frame_skip=args.frame_skip,
        seed=args.seed,
        device=args.device,
        subgoal_offsets=parse_subgoal_offsets(args.subgoal_offsets),
        reference_latent_pool_size=args.reference_latent_pool_size,
        max_cov_dim_for_json=args.max_cov_dim_for_json,
        save_json=args.save_json,
        save_npz=args.save_npz,
        append_tsv=args.append_tsv,
    )

    result = run_diagnostic(cfg)
    print("=== Diagnostic Summary ===")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
