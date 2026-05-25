from __future__ import annotations

from pathlib import Path
import sys

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
