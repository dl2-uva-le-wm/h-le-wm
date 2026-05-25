from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT_NAME = "runs"
REPRO_ROOT_NAME = "repro"


def stablewm_home() -> Path:
    env = os.environ.get("STABLEWM_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "data" / "stablewm").resolve()


def runs_root() -> Path:
    return stablewm_home() / RUNS_ROOT_NAME


def repro_root() -> Path:
    return stablewm_home() / REPRO_ROOT_NAME


def build_path_context() -> dict[str, Path]:
    return {
        "repo_root": REPO_ROOT,
        "stablewm_home": stablewm_home(),
        "run_root": runs_root(),
        "repro_root": repro_root(),
    }


def spec_output_root(*, root_kind: str, spec_slug: str, root_name: str | None = None) -> Path:
    roots = {
        RUNS_ROOT_NAME: runs_root(),
        REPRO_ROOT_NAME: repro_root(),
    }
    if root_kind not in roots:
        raise ValueError(f"Unsupported output root kind '{root_kind}'")
    name = (root_name or spec_slug).strip()
    if not name:
        raise ValueError("Spec output root name must not be empty")
    return roots[root_kind] / name


def resolve_output_path(raw: str | None, *, context: Mapping[str, Any] | None = None) -> Path | None:
    if not raw:
        return None
    format_context: dict[str, Any] = build_path_context()
    if context:
        format_context.update(dict(context))
    path = Path(str(raw).format(**format_context))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()
