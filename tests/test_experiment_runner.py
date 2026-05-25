from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h_le_wm.paths import REPO_ROOT, spec_output_root
from h_le_wm.experiments.run import (
    build_baseline_matrix_command,
    build_hierarchical_matrix_command,
    context_for_spec,
    load_index,
    load_index_entries,
    load_yaml,
    read_matrix_rows,
    resolve_spec_path,
    run_spec,
)

PUBLIC_SPEC_NAMES = {
    "matrix/pusht/baseline",
    "matrix/pusht/hierarchical",
    "matrix/cube/baseline",
    "matrix/cube/hierarchical",
    "smoke/pusht",
    "diagnostics/pusht/offline",
    "diagnostics/pusht/acting",
    "probe/pusht/phase_a/train",
    "probe/pusht/phase_b/train",
    "render/pusht/paper_diagnostics",
    "render/pusht/decoder_story_figures",
    "render/pusht/story_figures",
    "paper/reproduction",
}


def _run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "h_le_wm.experiments.run", *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        env=merged_env,
        check=False,
    )


def test_load_index_entries_exposes_machine_readable_registry():
    entries = load_index_entries()

    assert len(entries) == len(PUBLIC_SPEC_NAMES)
    assert all(set(("name", "kind", "operation", "path")).issubset(entry) for entry in entries)
    assert {entry["name"] for entry in entries} == PUBLIC_SPEC_NAMES


def test_load_index_maps_names_to_repo_relative_paths():
    index = load_index()

    assert index["matrix/pusht/baseline"] == "h_le_wm/experiments/specs/matrix/pusht/baseline.yaml"
    assert index["matrix/cube/hierarchical"] == "h_le_wm/experiments/specs/matrix/cube/hierarchical.yaml"
    assert index["smoke/pusht"] == "h_le_wm/experiments/specs/smoke/pusht.yaml"
    assert index["render/pusht/story_figures"] == "h_le_wm/experiments/specs/render/pusht/story_figures.yaml"
    assert index["paper/reproduction"] == "h_le_wm/experiments/specs/workflow/paper_reproduction.yaml"


def test_resolve_spec_path_uses_registered_names():
    path = resolve_spec_path("matrix/pusht/baseline")

    assert path == (REPO_ROOT / "h_le_wm/experiments/specs/matrix/pusht/baseline.yaml").resolve()


def test_all_public_specs_resolve_to_existing_files():
    for name in PUBLIC_SPEC_NAMES:
        assert resolve_spec_path(name).is_file(), name


def test_context_for_spec_injects_default_output_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path / "stablewm"))
    spec = load_yaml(resolve_spec_path("matrix/pusht/baseline"))

    context = context_for_spec(spec)

    assert context["spec_slug"] == "matrix__pusht__baseline"
    assert context["output_root"] == spec_output_root(
        root_kind="repro",
        spec_slug="matrix__pusht__baseline",
    )


def test_run_spec_dry_run_skips_checkpoint_presence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path / "stablewm"))
    spec = load_yaml(resolve_spec_path("matrix/pusht/baseline"))

    run_spec(spec, force=True, dry_run=True)

    captured = capsys.readouterr()
    assert "-m h_le_wm.eval.baseline_manifest" in captured.out
    assert not (tmp_path / "stablewm" / "repro").exists()


def test_build_baseline_matrix_command_uses_deterministic_output_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path / "stablewm"))
    spec = load_yaml(resolve_spec_path("matrix/pusht/baseline"))
    row = read_matrix_rows(REPO_ROOT / spec["sweep_csv"])[0]
    checkpoint = spec["checkpoints"][0]
    context = context_for_spec(spec)

    argv, result_path, manifest_path, meta = build_baseline_matrix_command(
        spec=spec,
        checkpoint=checkpoint,
        row=row,
        row_index=1,
        seed=42,
        context=context,
    )

    expected_root = tmp_path / "stablewm" / "repro" / "matrix__pusht__baseline" / "lewm" / "seed_042" / "row_001"
    assert result_path == expected_root / "pusht_results.txt"
    assert manifest_path == expected_root / "pusht_results_episodes.tsv"
    assert argv[:3] == [sys.executable, "-m", "h_le_wm.eval.baseline_manifest"]
    assert f"+output.root_dir={expected_root}" in argv
    assert meta["result_file"] == str(result_path)


def test_build_hierarchical_matrix_command_uses_deterministic_output_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path / "stablewm"))
    spec = load_yaml(resolve_spec_path("matrix/cube/hierarchical"))
    row = read_matrix_rows(REPO_ROOT / spec["sweep_csv"])[0]
    checkpoint = spec["checkpoints"][0]
    context = context_for_spec(spec)

    argv, result_path, manifest_path, meta = build_hierarchical_matrix_command(
        spec=spec,
        checkpoint=checkpoint,
        row=row,
        row_index=1,
        seed=42,
        context=context,
    )

    expected_root = tmp_path / "stablewm" / "repro" / "matrix__cube__hierarchical" / "hope2" / "seed_042" / "row_001"
    assert result_path == expected_root / "hi_cube_results.txt"
    assert manifest_path == expected_root / "hi_cube_results_episodes.tsv"
    assert argv[:3] == [sys.executable, "-m", "h_le_wm.eval.hierarchical"]
    assert f"+output.root_dir={expected_root}" in argv
    assert "planning.mode=hierarchical" in argv
    assert meta["result_file"] == str(result_path)


def test_smoke_spec_dry_run_prints_train_and_eval_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    stablewm_home = tmp_path / "stablewm"
    monkeypatch.setenv("STABLEWM_HOME", str(stablewm_home))

    result = _run_cli("--spec", "smoke/pusht", "--dry-run", env={"STABLEWM_HOME": str(stablewm_home)})

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "-m h_le_wm.train.hierarchical" in output
    assert "subdir=runs/pusht_smoke" in output
    assert "wandb.enabled=False" in output
    assert "-m h_le_wm.eval.hierarchical" in output
    assert f"+output.root_dir={stablewm_home / 'repro' / 'pusht_smoke'}" in output


def test_offline_and_acting_diagnostics_dry_runs_use_canonical_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stablewm_home = tmp_path / "stablewm"
    monkeypatch.setenv("STABLEWM_HOME", str(stablewm_home))

    offline = _run_cli(
        "--spec", "diagnostics/pusht/offline", "--dry-run", env={"STABLEWM_HOME": str(stablewm_home)}
    )
    acting = _run_cli(
        "--spec", "diagnostics/pusht/acting", "--dry-run", env={"STABLEWM_HOME": str(stablewm_home)}
    )

    offline_output = offline.stdout + offline.stderr
    acting_output = acting.stdout + acting.stderr
    assert offline.returncode == 0, offline_output
    assert acting.returncode == 0, acting_output
    assert "-m h_le_wm.experiments.pusht_diagnostics offline" in offline_output
    assert f"{stablewm_home / 'repro' / 'pusht_offline_diagnostics'}" in offline_output
    assert "-m h_le_wm.experiments.pusht_diagnostics acting" in acting_output
    assert f"{stablewm_home / 'repro' / 'pusht_acting_diagnostics'}" in acting_output


def test_probe_phase_specs_dry_run_force_wandb_off_and_stable_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stablewm_home = tmp_path / "stablewm"
    monkeypatch.setenv("STABLEWM_HOME", str(stablewm_home))

    phase_a = _run_cli(
        "--spec", "probe/pusht/phase_a/train", "--dry-run", env={"STABLEWM_HOME": str(stablewm_home)}
    )
    phase_b = _run_cli(
        "--spec", "probe/pusht/phase_b/train", "--dry-run", env={"STABLEWM_HOME": str(stablewm_home)}
    )

    phase_a_output = phase_a.stdout + phase_a.stderr
    phase_b_output = phase_b.stdout + phase_b.stderr
    assert phase_a.returncode == 0, phase_a_output
    assert phase_b.returncode == 0, phase_b_output
    assert "wandb.enabled=False" in phase_a_output
    assert "subdir=runs/pusht_probe_phase_a" in phase_a_output
    assert "probe.init_decoder_checkpoint=" in phase_b_output
    assert "runs/pusht_probe_phase_a/pusht_probe_phase_a_probe.pt" in phase_b_output
    assert "subdir=runs/pusht_probe_phase_b" in phase_b_output


def test_render_and_reproduction_dry_runs_reference_canonical_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stablewm_home = tmp_path / "stablewm"
    monkeypatch.setenv("STABLEWM_HOME", str(stablewm_home))

    render = _run_cli(
        "--spec", "render/pusht/paper_diagnostics", "--dry-run", env={"STABLEWM_HOME": str(stablewm_home)}
    )
    reproduction = _run_cli(
        "--spec", "paper/reproduction", "--dry-run", env={"STABLEWM_HOME": str(stablewm_home)}
    )

    render_output = render.stdout + render.stderr
    reproduction_output = reproduction.stdout + reproduction.stderr
    assert render.returncode == 0, render_output
    assert reproduction.returncode == 0, reproduction_output
    assert "-m h_le_wm.experiments.pusht_diagnostics offline" in render_output
    assert "scripts/render_hi_paper_diagnostics.py" in render_output
    assert f"{stablewm_home / 'repro' / 'matrix__pusht__baseline' / 'summary.csv'}" in render_output
    assert "-m h_le_wm.eval.baseline_manifest" in reproduction_output
    assert "-m h_le_wm.eval.hierarchical" in reproduction_output
    assert "scripts/render_hi_story_figures.py" in reproduction_output
