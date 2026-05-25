#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from hi_diagnostics import DiagnosticConfig, run_macro_action_manifold_diagnostic


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for the macro-action manifold diagnostic.\n"
            "This forwards to the shared hi_diagnostics implementation."
        )
    )
    p.add_argument("--policy", required=True, help="Policy path for AutoCostModel (relative to STABLEWM_HOME).")
    p.add_argument("--dataset-name", default="pusht_expert_train")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--chunk-len-tokens", type=int, default=5)
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
    p.add_argument("--save-npz", default=None, help="Optional output NPZ path.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = DiagnosticConfig(
        policy=args.policy,
        experiment_kind="macro_manifold",
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        img_size=args.img_size,
        num_eval_samples=args.num_eval_samples,
        num_empirical_chunks=args.num_empirical_chunks,
        goal_offset_steps=max(1, int(args.chunk_len_tokens) * 5),
        high_horizon=1,
        low_horizon=2,
        high_num_samples=args.cem_samples,
        high_iters=args.cem_iters,
        high_topk=0,
        low_num_samples=300,
        low_iters=30,
        low_topk=150,
        cem_elite_frac=args.cem_elite_frac,
        cem_bound_mode=args.cem_bound_mode,
        frame_skip=5,
        seed=args.seed,
        device=args.device,
        save_json=args.save_json,
        save_npz=args.save_npz,
    )
    result = run_macro_action_manifold_diagnostic(cfg)
    print("=== Macro-Action Manifold Diagnostic ===")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
