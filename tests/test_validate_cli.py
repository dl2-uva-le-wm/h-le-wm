from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h_le_wm.validate as validate_mod


def test_checkpoints_command_passes_tier_and_repeated_flags(monkeypatch):
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        validate_mod,
        "check_registered_checkpoints",
        lambda *, tier, checkpoint_names: seen.update({"tier": tier, "checkpoint_names": checkpoint_names}),
    )
    monkeypatch.setattr(
        validate_mod,
        "check_spec_checkpoints",
        lambda specs: seen.update({"specs": specs}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "checkpoints",
            "--tier",
            "supported-first-class",
            "--checkpoint",
            "baseline/pusht/lewm",
            "--checkpoint",
            "probe/pusht/phase_a",
            "--spec",
            "matrix/pusht/baseline",
            "--spec",
            "matrix/cube/hierarchical",
        ],
    )

    assert validate_mod.main() == 0
    assert seen == {
        "tier": "supported-first-class",
        "checkpoint_names": ["baseline/pusht/lewm", "probe/pusht/phase_a"],
        "specs": ["matrix/pusht/baseline", "matrix/cube/hierarchical"],
    }


def test_preflight_default_runs_required_now_paper_checks(monkeypatch):
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(validate_mod, "check_env", lambda: calls.append(("env", None)))
    monkeypatch.setattr(validate_mod, "check_baseline", lambda: calls.append(("baseline", None)))
    monkeypatch.setattr(validate_mod, "check_datasets", lambda datasets: calls.append(("datasets", datasets)))
    monkeypatch.setattr(
        validate_mod,
        "check_registered_checkpoints",
        lambda *, tier, checkpoint_names: calls.append(("checkpoints", (tier, checkpoint_names))),
    )
    monkeypatch.setattr(
        validate_mod,
        "check_spec_checkpoints",
        lambda specs: calls.append(("specs", specs)),
    )
    monkeypatch.setattr(sys, "argv", ["validate", "preflight"])

    assert validate_mod.main() == 0
    assert calls == [
        ("env", None),
        ("baseline", None),
        ("datasets", ["pusht", "cube"]),
        ("checkpoints", ("required-now", [])),
    ]


def test_outputs_command_passes_spec_name(monkeypatch):
    seen: dict[str, object] = {}

    monkeypatch.setattr(validate_mod, "check_spec_outputs", lambda spec_name: seen.update({"spec_name": spec_name}))
    monkeypatch.setattr(sys, "argv", ["validate", "outputs", "--spec", "paper/reproduction"])

    assert validate_mod.main() == 0
    assert seen == {"spec_name": "paper/reproduction"}


def _materialize_expected_outputs(paths: list[tuple[str, Path]]) -> None:
    for _, path in paths:
        if path.suffix or path.name in {"COMPLETE", "WORKFLOW_COMPLETE", "TRAIN_COMPLETE"}:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok\n")
            continue
        path.mkdir(parents=True, exist_ok=True)


def test_check_spec_outputs_validates_recursive_paper_reproduction(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path / "stablewm"))
    expected = validate_mod.iter_expected_output_paths("paper/reproduction")

    _materialize_expected_outputs(expected)

    validate_mod.check_spec_outputs("paper/reproduction")


def test_check_spec_outputs_validates_recursive_from_scratch(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path / "stablewm"))
    expected = validate_mod.iter_expected_output_paths("paper/from_scratch")

    _materialize_expected_outputs(expected)

    validate_mod.check_spec_outputs("paper/from_scratch")


def test_check_spec_outputs_reports_missing_artifact(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path / "stablewm"))
    expected = validate_mod.iter_expected_output_paths("paper/from_scratch")

    _materialize_expected_outputs(expected[1:])

    with pytest.raises(FileNotFoundError, match="Missing expected output"):
        validate_mod.check_spec_outputs("paper/from_scratch")
