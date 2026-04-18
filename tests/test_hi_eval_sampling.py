from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest


def _load_sample_eval_row_indices():
    """Load helper from hi_eval.py without importing heavy runtime deps."""
    src_path = Path(__file__).resolve().parents[1] / "hi_eval.py"
    source = src_path.read_text()
    mod = ast.parse(source)

    chunks = []
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sample_eval_row_indices":
            chunks.append(ast.get_source_segment(source, node))

    ns = {"np": np}
    exec("\n\n".join(chunks), ns)
    return ns["sample_eval_row_indices"]


SAMPLE_EVAL_ROW_INDICES = _load_sample_eval_row_indices()


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
