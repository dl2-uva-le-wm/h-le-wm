from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


def _load_block_only_helpers():
    src_path = Path(__file__).resolve().parents[1] / "eval_dataset_metrics.py"
    source = src_path.read_text()
    mod = ast.parse(source)

    chunks = []
    keep_assigns = {
        "PUSHT_BLOCK_POS_TOL",
        "PUSHT_BLOCK_ANGLE_TOL",
    }
    for node in mod.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & keep_assigns:
                chunks.append(ast.get_source_segment(source, node))
        if isinstance(node, ast.FunctionDef) and node.name == "compute_pusht_block_only_success":
            chunks.append(ast.get_source_segment(source, node))

    ns = {"np": np, "Sequence": tuple}
    exec("\n\n".join(chunks), ns)
    return (
        ns["compute_pusht_block_only_success"],
        ns["PUSHT_BLOCK_POS_TOL"],
        ns["PUSHT_BLOCK_ANGLE_TOL"],
    )


COMPUTE_PUSHT_BLOCK_ONLY_SUCCESS, POS_TOL, ANGLE_TOL = _load_block_only_helpers()


def test_block_only_success_accepts_matching_pose():
    block_pose = np.array([100.0, 200.0, 0.3])
    goal_pose = np.array([100.0, 200.0, 0.3])
    success = COMPUTE_PUSHT_BLOCK_ONLY_SUCCESS(block_pose, goal_pose)
    assert bool(success)


def test_block_only_success_rejects_large_position_error():
    block_pose = np.array([100.0, 200.0, 0.0])
    goal_pose = np.array([100.0 + POS_TOL + 1.0, 200.0, 0.0])
    success = COMPUTE_PUSHT_BLOCK_ONLY_SUCCESS(block_pose, goal_pose)
    assert not bool(success)


def test_block_only_success_handles_wrapped_angles():
    block_pose = np.array([100.0, 200.0, 2 * np.pi - 0.05])
    goal_pose = np.array([100.0, 200.0, 0.05])
    success = COMPUTE_PUSHT_BLOCK_ONLY_SUCCESS(block_pose, goal_pose)
    assert bool(success)


def test_block_only_success_vectorizes_over_batch():
    block_pose = np.array(
        [
            [10.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
        ]
    )
    goal_pose = np.array(
        [
            [10.0, 10.0, 0.0],
            [10.0, 10.0, ANGLE_TOL + 0.01],
        ]
    )
    success = COMPUTE_PUSHT_BLOCK_ONLY_SUCCESS(block_pose, goal_pose)
    np.testing.assert_array_equal(success, np.array([True, False]))
