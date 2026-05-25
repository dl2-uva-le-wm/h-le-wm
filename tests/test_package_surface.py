from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def test_canonical_modules_resolve_from_package_namespace():
    from h_le_wm.models.latent_action import LatentActionEncoder
    from h_le_wm.models.waypoint_sampling import sample_waypoints

    assert LatentActionEncoder.__module__ == "h_le_wm.models.latent_action"
    assert sample_waypoints.__module__ == "h_le_wm.models.waypoint_sampling"


def test_wrapper_help_surfaces_are_available():
    commands = [
        ("bash", "scripts/setup_baseline_checkpoints.sh", "--help"),
        ("bash", "scripts/setup_paper_datasets.sh", "--help"),
        ("bash", "scripts/setup_checkpoints.sh", "--help"),
        ("bash", "scripts/validate_preflight.sh", "--help"),
        ("bash", "scripts/run_pusht_smoke.sh", "--help"),
        ("bash", "scripts/run_pusht_baseline_matrix.sh", "--help"),
        ("bash", "scripts/run_pusht_hierarchical_matrix.sh", "--help"),
        ("bash", "scripts/run_cube_baseline_matrix.sh", "--help"),
        ("bash", "scripts/run_cube_hierarchical_matrix.sh", "--help"),
        ("bash", "scripts/run_pusht_offline_diagnostics.sh", "--help"),
        ("bash", "scripts/run_pusht_acting_diagnostics.sh", "--help"),
        ("bash", "scripts/train_pusht_probe_phase_a.sh", "--help"),
        ("bash", "scripts/train_pusht_probe_phase_b.sh", "--help"),
        ("bash", "scripts/render_pusht_paper_diagnostics.sh", "--help"),
        ("bash", "scripts/render_pusht_decoder_story_figures.sh", "--help"),
        ("bash", "scripts/render_pusht_story_figures.sh", "--help"),
        ("bash", "scripts/run_paper_reproduction.sh", "--help"),
    ]
    for command in commands:
        result = _run(*command)
        assert result.returncode == 0, result.stderr


def test_canonical_hierarchical_entrypoints_expose_help():
    pytest.importorskip("hydra")

    commands = [
        (sys.executable, "-m", "h_le_wm.eval.hierarchical", "--help"),
        (sys.executable, "-m", "h_le_wm.train.hierarchical", "--help"),
    ]
    for command in commands:
        result = _run(*command)
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "usage:" in output.lower()
