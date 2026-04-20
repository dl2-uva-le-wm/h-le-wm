from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PUSHT_BLOCK_POS_TOL = 20.0
PUSHT_BLOCK_ANGLE_TOL = np.pi / 9


def compute_pusht_block_only_success(
    block_pose: np.ndarray | Sequence[float],
    goal_pose: np.ndarray | Sequence[float],
    *,
    pos_tol: float = PUSHT_BLOCK_POS_TOL,
    angle_tol: float = PUSHT_BLOCK_ANGLE_TOL,
) -> np.ndarray:
    """Return PushT success while ignoring agent position.

    This matches the upstream PushT success rule, except it only checks block
    position and block angle instead of the full `[agent, block]` state.
    """
    block_pose_arr = np.asarray(block_pose, dtype=np.float64)
    goal_pose_arr = np.asarray(goal_pose, dtype=np.float64)

    if block_pose_arr.shape[-1] < 3 or goal_pose_arr.shape[-1] < 3:
        raise ValueError("block_pose and goal_pose must have at least 3 values.")

    pos_diff = np.linalg.norm(goal_pose_arr[..., :2] - block_pose_arr[..., :2], axis=-1)
    angle_diff = np.abs(goal_pose_arr[..., 2] - block_pose_arr[..., 2])
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    return np.logical_and(pos_diff < pos_tol, angle_diff < angle_tol)


def evaluate_from_dataset_with_optional_metrics(
    *,
    world: Any,
    dataset: Any,
    episodes_idx: Sequence[int],
    start_steps: Sequence[int],
    goal_offset_steps: int,
    eval_budget: int,
    callables: list[dict] | None = None,
    save_video: bool = True,
    video_path: str | Path = "./",
    enable_pusht_block_only: bool = False,
) -> dict[str, Any]:
    """Mirror `stable_worldmodel.World.evaluate_from_dataset` with extra metrics.

    The rollout behavior intentionally matches upstream evaluation so any added
    metrics are post-hoc and do not affect planning or episode execution.
    """
    assert (
        world.envs.envs[0].spec.max_episode_steps is None
        or world.envs.envs[0].spec.max_episode_steps >= goal_offset_steps
    ), "env max_episode_steps must be greater than eval_budget"

    ep_idx_arr = np.asarray(episodes_idx)
    start_steps_arr = np.asarray(start_steps)
    end_steps = start_steps_arr + goal_offset_steps + 1

    if len(ep_idx_arr) != len(start_steps_arr):
        raise ValueError("episodes_idx and start_steps must have the same length")
    if len(ep_idx_arr) != world.num_envs:
        raise ValueError("Number of episodes to evaluate must match number of envs")

    data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)
    columns = dataset.column_names

    init_step_per_env: dict[str, list[Any]] = defaultdict(list)
    goal_step_per_env: dict[str, list[Any]] = defaultdict(list)
    for ep in data:
        for col in columns:
            if col.startswith("goal"):
                continue
            if col.startswith("pixels"):
                ep[col] = ep[col].permute(0, 2, 3, 1)
            if not isinstance(ep[col], (torch.Tensor, np.ndarray)):
                continue

            init_data = ep[col][0]
            goal_data = ep[col][-1]
            if not isinstance(init_data, (np.ndarray, torch.Tensor)):
                continue

            if isinstance(init_data, torch.Tensor):
                init_data = init_data.numpy()
            if isinstance(goal_data, torch.Tensor):
                goal_data = goal_data.numpy()

            init_step_per_env[col].append(init_data)
            goal_step_per_env[col].append(goal_data)

    init_step = {k: np.stack(v) for k, v in deepcopy(init_step_per_env).items()}
    goal_step = {}
    for key, value in goal_step_per_env.items():
        goal_key = "goal" if key == "pixels" else f"goal_{key}"
        goal_step[goal_key] = np.stack(value)

    seeds = init_step.get("seed")
    variations_dict = {
        k.removeprefix("variation."): v
        for k, v in init_step.items()
        if k.startswith("variation.")
    }
    options = [{} for _ in range(world.num_envs)]
    if variations_dict:
        for i in range(world.num_envs):
            options[i]["variation"] = list(variations_dict.keys())
            options[i]["variation_values"] = {k: v[i] for k, v in variations_dict.items()}

    init_step.update(deepcopy(goal_step))
    world.reset(seed=seeds, options=options)

    callables = callables or []
    for i, env in enumerate(world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        for spec in callables:
            method_name = spec["method"]
            if not hasattr(env_unwrapped, method_name):
                continue

            method = getattr(env_unwrapped, method_name)
            args = spec.get("args", spec)
            prepared_args = {}
            for args_name, args_data in args.items():
                value = args_data.get("value")
                in_dataset = args_data.get("in_dataset", True)
                if in_dataset:
                    if value not in init_step:
                        continue
                    prepared_args[args_name] = deepcopy(init_step[value][i])
                else:
                    prepared_args[args_name] = value
            method(**prepared_args)

    for i in range(world.num_envs):
        if "goal_state" in init_step and "goal_state" in goal_step:
            assert np.array_equal(init_step["goal_state"][i], goal_step["goal_state"][i]), (
                "Goal state info does not match at reset"
            )

    results: dict[str, Any] = {
        "success_rate": 0.0,
        "episode_successes": np.zeros(len(episodes_idx), dtype=bool),
        "seeds": seeds,
    }
    if enable_pusht_block_only:
        results["success_rate_block_only"] = 0.0
        results["episode_successes_block_only"] = np.zeros(len(episodes_idx), dtype=bool)

    shape_prefix = world.infos["pixels"].shape[:2]
    init_step = {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:]) for k, v in init_step.items()
    }
    goal_step = {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:]) for k, v in goal_step.items()
    }
    world.infos.update(deepcopy(init_step))
    world.infos.update(deepcopy(goal_step))

    if "goal" in goal_step and "goal" in world.infos:
        assert np.allclose(world.infos["goal"], goal_step["goal"]), "Goal info does not match"

    target_frames = torch.stack([ep["pixels"] for ep in data]).numpy()
    video_frames = np.empty(
        (world.num_envs, eval_budget, *world.infos["pixels"].shape[-3:]),
        dtype=np.uint8,
    )

    for step_idx in range(eval_budget):
        video_frames[:, step_idx] = world.infos["pixels"][:, -1]
        world.infos.update(deepcopy(goal_step))
        world.step()
        results["episode_successes"] = np.logical_or(
            results["episode_successes"],
            np.asarray(world.terminateds, dtype=bool),
        )
        if enable_pusht_block_only:
            if "block_pose" not in world.infos or "goal_pose" not in world.infos:
                raise ValueError(
                    "PushT block-only metric requires 'block_pose' and 'goal_pose' in env infos."
                )
            results["episode_successes_block_only"] = np.logical_or(
                results["episode_successes_block_only"],
                compute_pusht_block_only_success(
                    world.infos["block_pose"],
                    world.infos["goal_pose"],
                ),
            )

        world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))

    video_frames[:, -1] = world.infos["pixels"][:, -1]

    n_episodes = len(episodes_idx)
    results["success_rate"] = float(np.sum(results["episode_successes"])) / n_episodes * 100.0
    if enable_pusht_block_only:
        results["success_rate_block_only"] = (
            float(np.sum(results["episode_successes_block_only"])) / n_episodes * 100.0
        )

    if save_video:
        import imageio

        target_len = target_frames.shape[1]
        video_path_obj = Path(video_path)
        video_path_obj.mkdir(parents=True, exist_ok=True)
        for i in range(world.num_envs):
            out = imageio.get_writer(
                video_path_obj / f"rollout_{i}.mp4",
                fps=15,
                codec="libx264",
            )
            goals = np.vstack([target_frames[i, -1], target_frames[i, -1]])
            for t in range(eval_budget):
                stacked_frame = np.vstack([video_frames[i, t], target_frames[i, t % target_len]])
                frame = np.hstack([stacked_frame, goals])
                out.append_data(frame)
            out.close()
        print(f"Video saved to {video_path_obj}")

    if results["seeds"] is not None:
        assert np.unique(results["seeds"]).shape[0] == n_episodes, (
            "Some episode seeds are identical!"
        )

    return results
