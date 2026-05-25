from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h_le_wm.paths import (
    REPO_ROOT,
    REPRO_ROOT_NAME,
    RUNS_ROOT_NAME,
    build_path_context,
    repro_root,
    resolve_output_path,
    runs_root,
    spec_output_root,
    stablewm_home,
)


def test_stablewm_home_uses_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    override = tmp_path / "paper-home"
    monkeypatch.setenv("STABLEWM_HOME", str(override))

    assert stablewm_home() == override.resolve()


def test_stablewm_home_defaults_under_repo(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("STABLEWM_HOME", raising=False)

    assert stablewm_home() == (REPO_ROOT / "data" / "stablewm").resolve()


def test_runs_and_repro_roots_follow_stablewm_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    home = tmp_path / "stablewm-home"
    monkeypatch.setenv("STABLEWM_HOME", str(home))

    assert runs_root() == home.resolve() / RUNS_ROOT_NAME
    assert repro_root() == home.resolve() / REPRO_ROOT_NAME


def test_spec_output_root_uses_default_slug_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    home = tmp_path / "stablewm-home"
    monkeypatch.setenv("STABLEWM_HOME", str(home))

    assert spec_output_root(root_kind="repro", spec_slug="matrix__pusht__baseline") == (
        home.resolve() / "repro" / "matrix__pusht__baseline"
    )


def test_spec_output_root_accepts_explicit_root_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    home = tmp_path / "stablewm-home"
    monkeypatch.setenv("STABLEWM_HOME", str(home))

    assert spec_output_root(root_kind="runs", spec_slug="unused", root_name="pusht_smoke") == (
        home.resolve() / "runs" / "pusht_smoke"
    )


def test_resolve_output_path_uses_shared_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    home = tmp_path / "stablewm-home"
    monkeypatch.setenv("STABLEWM_HOME", str(home))

    context = build_path_context()
    context["output_root"] = spec_output_root(root_kind="runs", spec_slug="train__pusht__smoke")

    assert resolve_output_path("{output_root}/metrics.json", context=context) == (
        home.resolve() / "runs" / "train__pusht__smoke" / "metrics.json"
    )
