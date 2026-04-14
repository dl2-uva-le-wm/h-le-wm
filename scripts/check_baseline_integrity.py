#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SUBMODULE = REPO_ROOT / "third_party" / "lewm"
LOCK_FILE = REPO_ROOT / "BASELINE_LOCK.md"

# Root paths that must not be reintroduced from the baseline codebase.
BLOCKED_ROOT_PATHS = [
    "jepa.py",
    "module.py",
    "utils.py",
    "config/train/lewm.yaml",
    "config/train/data/pusht.yaml",
    "config/train/data/tworoom.yaml",
    "config/train/data/dmc.yaml",
    "config/train/data/ogb.yaml",
    "config/train/launcher/local.yaml",
    "config/eval/pusht.yaml",
    "config/eval/tworoom.yaml",
    "config/eval/reacher.yaml",
    "config/eval/cube.yaml",
    "config/eval/launcher/local.yaml",
    "config/eval/solver/cem.yaml",
    "config/eval/solver/adam.yaml",
]


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    out = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {out.stderr.strip()}")
    return out.stdout.strip()


def parse_locked_hash() -> str:
    if not LOCK_FILE.exists():
        raise RuntimeError("BASELINE_LOCK.md missing")
    text = LOCK_FILE.read_text()
    m = re.search(r"Pinned Commit:\s*`([0-9a-f]{40})`", text)
    if not m:
        raise RuntimeError("Could not parse pinned commit hash from BASELINE_LOCK.md")
    return m.group(1)


def check_submodule_present() -> None:
    if not BASELINE_SUBMODULE.exists():
        raise RuntimeError("Baseline submodule directory missing: third_party/lewm")
    gitmodules = REPO_ROOT / ".gitmodules"
    if not gitmodules.exists():
        raise RuntimeError(".gitmodules missing")
    gm = gitmodules.read_text()
    if "third_party/lewm" not in gm:
        raise RuntimeError(".gitmodules has no third_party/lewm entry")


def check_submodule_pointer(allow_pointer_update: bool) -> None:
    locked = parse_locked_hash()
    current = run(["git", "-C", str(BASELINE_SUBMODULE), "rev-parse", "HEAD"])
    if not allow_pointer_update and current != locked:
        raise RuntimeError(
            "Submodule pointer mismatch. "
            f"locked={locked}, current={current}. "
            "Use --allow-pointer-update only for intentional baseline bumps."
        )


def check_submodule_clean() -> None:
    status = run(["git", "-C", str(BASELINE_SUBMODULE), "status", "--porcelain", "--untracked-files=all"])
    noise_prefixes = (
        "?? __pycache__/",
        "?? .pytest_cache/",
        "?? .mypy_cache/",
        "?? .ruff_cache/",
    )
    lines = [
        line
        for line in status.splitlines()
        if line and not any(line.startswith(prefix) for prefix in noise_prefixes)
    ]
    if lines:
        listed = "\n".join(f"- {line}" for line in lines)
        raise RuntimeError(
            "Baseline submodule has local modifications. "
            "Reset or commit changes in third_party/lewm before proceeding.\n"
            f"{listed}"
        )


def check_blocked_root_paths() -> None:
    offenders = []
    for rel in BLOCKED_ROOT_PATHS:
        if (REPO_ROOT / rel).exists():
            offenders.append(rel)
    if offenders:
        listed = "\n".join(f"- {x}" for x in offenders)
        raise RuntimeError(
            "Baseline-owned files were reintroduced in root:\n"
            f"{listed}\n"
            "Keep baseline code/config only in third_party/lewm."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check baseline submodule integrity")
    parser.add_argument(
        "--allow-pointer-update",
        action="store_true",
        help="Allow current submodule HEAD to differ from locked hash (for intentional baseline bump PRs)",
    )
    args = parser.parse_args()

    try:
        check_submodule_present()
        check_submodule_pointer(args.allow_pointer_update)
        check_submodule_clean()
        check_blocked_root_paths()
    except Exception as exc:  # noqa: BLE001
        print(f"[baseline-integrity] FAIL: {exc}")
        return 1

    print("[baseline-integrity] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
