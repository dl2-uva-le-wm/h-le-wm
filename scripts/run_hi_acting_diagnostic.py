#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from hi_acting_diagnostics import ActingDiagnosticConfig, run_acting_diagnostic


def parse_subgoal_offsets(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return (2, 3, 5)
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return tuple(values) if values else (2, 3, 5)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Online acting diagnostics for hierarchical PushT checkpoints.")
    p.add_argument("--policy", required=True, help="Policy path for AutoCostModel (relative to STABLEWM_HOME).")
    p.add_argument(
        "--experiment-kind",
        required=True,
        choices=[
            "oracle_subgoal_acting",
            "low_level_reality_gap",
            "generated_subgoal_acting",
            "online_hierarchical_logging",
        ],
    )
    p.add_argument("--dataset-name", default="pusht_expert_train")
    p.add_argument("--eval-config", default="h_le_wm/config/eval/hi_pusht.yaml")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--goal-offset-steps", type=int, default=50)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--high-horizon", type=int, default=2)
    p.add_argument("--low-horizon", type=int, default=2)
    p.add_argument("--low-receding-horizon", type=int, default=1)
    p.add_argument("--high-num-samples", type=int, default=1500)
    p.add_argument("--high-iters", type=int, default=40)
    p.add_argument("--high-topk", type=int, default=10)
    p.add_argument("--low-num-samples", type=int, default=900)
    p.add_argument("--low-iters", type=int, default=20)
    p.add_argument("--low-topk", type=int, default=150)
    p.add_argument("--frame-skip", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--subgoal-offsets", default="2,3,5")
    p.add_argument("--num-reference-samples", type=int, default=4096)
    p.add_argument("--save-json", default=None)
    p.add_argument("--save-npz", default=None)
    p.add_argument("--append-tsv", default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = ActingDiagnosticConfig(
        policy=args.policy,
        experiment_kind=args.experiment_kind,
        dataset_name=args.dataset_name,
        eval_config=args.eval_config,
        cache_dir=args.cache_dir,
        img_size=args.img_size,
        num_eval=args.num_eval,
        goal_offset_steps=args.goal_offset_steps,
        eval_budget=args.eval_budget,
        high_horizon=args.high_horizon,
        low_horizon=args.low_horizon,
        low_receding_horizon=args.low_receding_horizon,
        high_num_samples=args.high_num_samples,
        high_iters=args.high_iters,
        high_topk=args.high_topk,
        low_num_samples=args.low_num_samples,
        low_iters=args.low_iters,
        low_topk=args.low_topk,
        frame_skip=args.frame_skip,
        seed=args.seed,
        device=args.device,
        subgoal_offsets=parse_subgoal_offsets(args.subgoal_offsets),
        num_reference_samples=args.num_reference_samples,
        save_json=args.save_json,
        save_npz=args.save_npz,
        append_tsv=args.append_tsv,
    )
    result = run_acting_diagnostic(cfg)
    print("=== Acting Diagnostic Summary ===")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
