from __future__ import annotations

import ast
from pathlib import Path
import re

import numpy as np
import pytest


def _load_hi_eval_helpers():
    """Load selected helpers from hi_eval.py without importing heavy runtime deps."""
    src_path = Path(__file__).resolve().parents[1] / "hi_eval.py"
    source = src_path.read_text()
    mod = ast.parse(source)

    wanted = {
        "sample_eval_row_indices",
        "extract_eval_index_from_video_name",
        "map_eval_video_paths",
        "extract_episode_successes",
        "build_episode_outcomes",
        "format_outcome_line",
    }
    chunks = []
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            chunks.append(ast.get_source_segment(source, node))

    ns = {"np": np, "Path": Path, "re": re}
    exec("\n\n".join(chunks), ns)
    return ns


_HI_EVAL_NS = _load_hi_eval_helpers()
SAMPLE_EVAL_ROW_INDICES = _HI_EVAL_NS["sample_eval_row_indices"]
MAP_EVAL_VIDEO_PATHS = _HI_EVAL_NS["map_eval_video_paths"]
EXTRACT_EPISODE_SUCCESSES = _HI_EVAL_NS["extract_episode_successes"]
BUILD_EPISODE_OUTCOMES = _HI_EVAL_NS["build_episode_outcomes"]
FORMAT_OUTCOME_LINE = _HI_EVAL_NS["format_outcome_line"]


def test_sample_eval_row_indices_uses_full_population():
    valid_indices = np.array([10, 20, 30, 40], dtype=np.int64)
    sampled = SAMPLE_EVAL_ROW_INDICES(valid_indices, num_eval=4, seed=0)
    assert sampled.shape == (4,)
    np.testing.assert_array_equal(np.sort(sampled), valid_indices)


def test_sample_eval_row_indices_deterministic_for_seed():
    valid_indices = np.arange(100, 160, dtype=np.int64)
    a = SAMPLE_EVAL_ROW_INDICES(valid_indices, num_eval=20, seed=123)
    b = SAMPLE_EVAL_ROW_INDICES(valid_indices, num_eval=20, seed=123)
    np.testing.assert_array_equal(a, b)


def test_sample_eval_row_indices_raises_when_insufficient():
    valid_indices = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="Not enough valid starting points"):
        SAMPLE_EVAL_ROW_INDICES(valid_indices, num_eval=4, seed=0)


def test_map_eval_video_paths_prefers_index_in_filename():
    video_files = [
        Path("/tmp/run/summary.mp4"),
        Path("/tmp/run/eval_1.mp4"),
        Path("/tmp/run/eval_0.mp4"),
    ]
    mapped = MAP_EVAL_VIDEO_PATHS(video_files=video_files, num_eval=2)
    assert mapped == ["/tmp/run/eval_0.mp4", "/tmp/run/eval_1.mp4"]


def test_build_episode_outcomes_formats_pass_fail_and_video_path():
    outcomes = BUILD_EPISODE_OUTCOMES(
        eval_episodes=np.array([10, 11], dtype=np.int64),
        eval_start_idx=np.array([100, 150], dtype=np.int64),
        episode_successes=np.array([True, False]),
        eval_video_paths=["/tmp/run/eval_0.mp4", "/tmp/run/eval_1.mp4"],
    )
    assert outcomes[0]["status"] == "PASS"
    assert outcomes[1]["status"] == "FAIL"
    line = FORMAT_OUTCOME_LINE(outcomes[1])
    assert "status" not in line
    assert "FAIL" in line
    assert "eval_index=1" in line
    assert "episode_id=11" in line
    assert "start_step=150" in line
    assert "video_path=/tmp/run/eval_1.mp4" in line


def test_extract_episode_successes_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="Mismatch between sampled evaluations"):
        EXTRACT_EPISODE_SUCCESSES(
            metrics={"episode_successes": [True, False]},
            expected_count=3,
        )
