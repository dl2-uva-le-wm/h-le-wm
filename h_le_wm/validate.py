from __future__ import annotations

import argparse
import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path

from h_le_wm.checkpoints import KNOWN_TIERS, iter_registry_entries, resolve_checkpoint_target
from h_le_wm.experiments.run import context_for_spec, load_yaml, resolve_spec_path
from h_le_wm.paths import REPO_ROOT, stablewm_home


PAPER_DATASET_FILES = {
    "pusht": "pusht_expert_train.h5",
    "cube": "cube_single_expert.h5",
}
DEFAULT_DATASETS = ("pusht", "cube")


def check_env() -> None:
    required = ["stable-worldmodel", "stable-pretraining", "torch", "numpy", "pyyaml"]
    missing = []
    for name in required:
        try:
            print(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing required packages: {', '.join(missing)}")


def check_baseline() -> None:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "tools" / "check_baseline_integrity.py")]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError("Baseline integrity check failed.")


def check_datasets(dataset_keys: list[str]) -> None:
    home = stablewm_home()
    for key in dataset_keys:
        if key not in PAPER_DATASET_FILES:
            raise ValueError(f"Unsupported dataset key '{key}'")
        path = home / PAPER_DATASET_FILES[key]
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        print(f"[ok] {path}")


def parse_dataset_keys(raw: str) -> list[str]:
    keys = [item.strip() for item in str(raw).split(",") if item.strip()]
    return keys or list(DEFAULT_DATASETS)


def resolve_spec_checkpoint_paths(spec_name: str) -> list[Path]:
    spec = load_yaml(resolve_spec_path(spec_name))
    requirements = spec.get("requirements", {})
    if not isinstance(requirements, dict):
        return []
    checkpoints = requirements.get("checkpoints", [])
    if not checkpoints:
        return []
    context = context_for_spec(spec)
    resolved: list[Path] = []
    for raw in checkpoints:
        resolved.append(Path(str(raw).format(**context)))
    return resolved


def check_spec_checkpoints(spec_names: list[str]) -> None:
    for spec_name in spec_names:
        for path in resolve_spec_checkpoint_paths(spec_name):
            if not path.exists():
                raise FileNotFoundError(f"Missing checkpoint for spec '{spec_name}': {path}")
            print(f"[ok] spec:{spec_name} -> {path}")


def check_registered_checkpoints(*, tier: str | None, checkpoint_names: list[str]) -> None:
    entries = iter_registry_entries(tier=tier, names=checkpoint_names)
    for entry in entries:
        path = resolve_checkpoint_target(entry)
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint '{entry['name']}': {path}")
        print(f"[ok] registry:{entry['name']} -> {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the H-LeWM paper-ready workspace.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="Check the canonical Python package surface.")
    sub.add_parser("baseline", help="Check baseline submodule integrity.")

    datasets = sub.add_parser("datasets", help="Check required dataset files.")
    datasets.add_argument("--datasets", default="pusht,cube")

    checkpoints = sub.add_parser("checkpoints", help="Check registry checkpoints and optional spec requirements.")
    checkpoints.add_argument("--tier", choices=KNOWN_TIERS, default="required-now")
    checkpoints.add_argument("--checkpoint", action="append", default=[])
    checkpoints.add_argument("--spec", action="append", default=[])

    preflight = sub.add_parser("preflight", help="Run the non-expensive paper preflight checks.")
    preflight.add_argument("--datasets", default="pusht,cube")
    preflight.add_argument("--tier", choices=KNOWN_TIERS, default="required-now")
    preflight.add_argument("--checkpoint", action="append", default=[])
    preflight.add_argument("--spec", action="append", default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "env":
        check_env()
        return 0
    if args.command == "baseline":
        check_baseline()
        return 0
    if args.command == "datasets":
        check_datasets(parse_dataset_keys(args.datasets))
        return 0
    if args.command == "checkpoints":
        check_registered_checkpoints(tier=args.tier or None, checkpoint_names=list(args.checkpoint))
        if args.spec:
            check_spec_checkpoints(list(args.spec))
        return 0
    if args.command == "preflight":
        check_env()
        check_baseline()
        check_datasets(parse_dataset_keys(args.datasets))
        check_registered_checkpoints(tier=args.tier or None, checkpoint_names=list(args.checkpoint))
        if args.spec:
            check_spec_checkpoints(list(args.spec))
        return 0
    raise RuntimeError(f"Unsupported validate command '{args.command}'")


if __name__ == "__main__":
    raise SystemExit(main())
