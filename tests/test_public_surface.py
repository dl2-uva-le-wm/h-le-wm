from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

DOC_PATHS = [
    ROOT / "docs" / "install.md",
    ROOT / "docs" / "datasets.md",
    ROOT / "docs" / "reproduction.md",
    ROOT / "docs" / "hardware.md",
    ROOT / "docs" / "experiments.md",
    ROOT / "docs" / "checkpoints.md",
    ROOT / "docs" / "outputs.md",
    ROOT / "docs" / "architecture.md",
    ROOT / "docs" / "architecture_variants.md",
]

CANONICAL_SURFACE_PATHS = [
    ROOT / "README.md",
    ROOT / "h_le_wm" / "experiments" / "index.yaml",
    *sorted((ROOT / "h_le_wm" / "experiments" / "specs").rglob("*.yaml")),
    ROOT / "scripts" / "setup_baseline_checkpoints.sh",
    ROOT / "scripts" / "run_pusht_smoke.sh",
    ROOT / "scripts" / "run_pusht_baseline_matrix.sh",
    ROOT / "scripts" / "run_pusht_hierarchical_matrix.sh",
    ROOT / "scripts" / "run_cube_baseline_matrix.sh",
    ROOT / "scripts" / "run_cube_hierarchical_matrix.sh",
    ROOT / "scripts" / "run_pusht_offline_diagnostics.sh",
    ROOT / "scripts" / "run_pusht_acting_diagnostics.sh",
    ROOT / "scripts" / "train_pusht_probe_phase_a.sh",
    ROOT / "scripts" / "train_pusht_probe_phase_b.sh",
    ROOT / "scripts" / "render_pusht_paper_diagnostics.sh",
    ROOT / "scripts" / "render_pusht_decoder_story_figures.sh",
    ROOT / "scripts" / "render_pusht_story_figures.sh",
    ROOT / "scripts" / "run_paper_reproduction.sh",
]

ROOT_CONFIG_RE = re.compile(r"(^|[^A-Za-z0-9_./-])config/")


def test_required_public_docs_exist():
    for path in DOC_PATHS:
        assert path.is_file(), path


def test_readme_links_all_required_docs():
    readme = (ROOT / "README.md").read_text()

    for path in DOC_PATHS:
        assert path.name in readme, path.name


def test_canonical_surface_avoids_root_config_paths():
    for path in CANONICAL_SURFACE_PATHS:
        text = path.read_text()
        assert ROOT_CONFIG_RE.search(text) is None, path
