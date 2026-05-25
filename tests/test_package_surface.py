from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXEC_ONLY_WRAPPERS = [
    "scripts/setup_baseline_checkpoints.sh",
    "scripts/setup_checkpoints.sh",
    "scripts/validate_preflight.sh",
    "scripts/run_pusht_smoke.sh",
    "scripts/run_pusht_baseline_matrix.sh",
    "scripts/run_pusht_hierarchical_matrix.sh",
    "scripts/run_cube_baseline_matrix.sh",
    "scripts/run_cube_hierarchical_matrix.sh",
    "scripts/run_pusht_offline_diagnostics.sh",
    "scripts/run_pusht_acting_diagnostics.sh",
    "scripts/train_pusht_probe_phase_a.sh",
    "scripts/train_pusht_probe_phase_b.sh",
    "scripts/render_pusht_paper_diagnostics.sh",
    "scripts/render_pusht_decoder_story_figures.sh",
    "scripts/render_pusht_story_figures.sh",
    "scripts/run_paper_reproduction.sh",
]

SOURCEABLE_WRAPPERS = [
    "scripts/setup_datasets.sh",
    "scripts/setup_paper_datasets.sh",
]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def _run_shell(shell: str, command: str) -> subprocess.CompletedProcess[str]:
    return _run(shell, "-lc", command)


def test_canonical_modules_resolve_from_package_namespace():
    from h_le_wm.models.latent_action import LatentActionEncoder
    from h_le_wm.models.waypoint_sampling import sample_waypoints

    assert LatentActionEncoder.__module__ == "h_le_wm.models.latent_action"
    assert sample_waypoints.__module__ == "h_le_wm.models.waypoint_sampling"


def test_wrapper_help_surfaces_are_available():
    for wrapper in [*EXEC_ONLY_WRAPPERS, "scripts/setup_paper_datasets.sh"]:
        result = _run("bash", wrapper, "--help")
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("shell", [shell for shell in ("bash", "zsh") if shutil.which(shell)])
def test_sourceable_dataset_wrappers_support_bash_and_zsh(shell: str):
    for wrapper in SOURCEABLE_WRAPPERS:
        result = _run_shell(shell, f"source {wrapper} --help")
        output = result.stdout + result.stderr
        assert result.returncode == 0, output


@pytest.mark.parametrize("shell", [shell for shell in ("bash", "zsh") if shutil.which(shell)])
def test_exec_only_wrappers_reject_sourcing(shell: str):
    for wrapper in EXEC_ONLY_WRAPPERS:
        result = _run_shell(shell, f"source {wrapper}")
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "Do not source" in output


@pytest.mark.parametrize("shell", [shell for shell in ("bash", "zsh") if shutil.which(shell)])
def test_dataset_lists_split_consistently_in_bash_and_zsh(shell: str):
    result = _run_shell(
        shell,
        'tmpdir="$(mktemp -d)"; source scripts/setup_datasets.sh --home "$tmpdir" --datasets pusht,cube,invalid',
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Unsupported dataset key: 'invalid'" in output
    assert "pusht cube" not in output


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
