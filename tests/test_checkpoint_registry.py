from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h_le_wm.checkpoints import (
    HF_BASELINE_SOURCES,
    KNOWN_TIERS,
    fetch_hf_baselines,
    iter_registry_entries,
    load_checkpoint_registry,
    parse_checkpoint_assignment,
    resolve_checkpoint_target,
)


def test_checkpoint_registry_has_expected_shape_and_names():
    entries = load_checkpoint_registry()
    by_name = {entry["name"]: entry["relpath"] for entry in entries}

    assert len(entries) >= 6
    assert {entry["name"] for entry in entries} >= {
        "baseline/pusht/lewm",
        "baseline/cube/lewm",
        "hierarchical/pusht/default_epoch15",
        "hierarchical/cube/default_epoch15",
        "probe/pusht/phase_a",
        "probe/pusht/phase_b",
    }
    assert by_name["hierarchical/pusht/default_epoch15"] == (
        "runs/pusht_hierarchical_default/pusht_hierarchical_default_epoch_15_object.ckpt"
    )
    assert by_name["hierarchical/cube/default_epoch15"] == (
        "runs/cube_hierarchical_default/cube_hierarchical_default_epoch_15_object.ckpt"
    )
    assert by_name["probe/pusht/phase_a"] == "runs/pusht_probe_phase_a/pusht_probe_phase_a_probe.pt"
    assert by_name["probe/pusht/phase_b"] == "runs/pusht_probe_phase_b/pusht_probe_phase_b_probe.pt"
    assert all(entry["relpath"] for entry in entries)
    assert all(entry["tiers"] for entry in entries)


def test_checkpoint_targets_resolve_under_stablewm_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path / "stablewm"))
    entry = next(item for item in load_checkpoint_registry() if item["name"] == "baseline/pusht/lewm")

    target = resolve_checkpoint_target(entry)

    assert target == (tmp_path / "stablewm" / "pusht" / "lewm_object.ckpt").resolve()


def test_supported_first_class_tier_includes_all_named_public_families():
    entries = iter_registry_entries(tier="supported-first-class")

    assert {entry["name"] for entry in entries} >= {
        "baseline/pusht/lewm",
        "baseline/cube/lewm",
        "hierarchical/pusht/default_epoch15",
        "hierarchical/cube/default_epoch15",
        "probe/pusht/phase_a",
        "probe/pusht/phase_b",
    }
    assert "required-now" in KNOWN_TIERS


def test_duplicate_checkpoint_names_are_rejected(tmp_path: Path):
    registry_path = tmp_path / "checkpoint_registry.yaml"
    registry_path.write_text(
        """
checkpoints:
  - name: baseline/pusht/lewm
    relpath: pusht/lewm_object.ckpt
    tiers: [required-now]
  - name: baseline/pusht/lewm
    relpath: cube/lewm_object.ckpt
    tiers: [supported-first-class]
""".strip()
    )

    with pytest.raises(ValueError, match="Duplicate checkpoint registry name"):
        load_checkpoint_registry(path=registry_path)


def test_docs_placeholder_checkpoint_path_is_rejected_early():
    with pytest.raises(FileNotFoundError, match="docs placeholder path"):
        parse_checkpoint_assignment("baseline/pusht/lewm=/absolute/path/to/pusht_lewm_object.ckpt")


def test_hf_baseline_sources_cover_required_now_registry_names():
    required_now = {entry["name"] for entry in iter_registry_entries(tier="required-now")}
    assert required_now <= set(HF_BASELINE_SOURCES)


def test_fetch_hf_baselines_dry_run_returns_canonical_targets(tmp_path: Path):
    targets = fetch_hf_baselines(home=tmp_path, dry_run=True, force=False, allow_non_strict=False)

    assert targets == [
        (tmp_path / "pusht" / "lewm_object.ckpt").resolve(),
        (tmp_path / "cube" / "lewm_object.ckpt").resolve(),
    ]
