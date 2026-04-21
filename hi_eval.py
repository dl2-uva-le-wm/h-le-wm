from __future__ import annotations

import os
import time
from pathlib import Path
import re

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import baseline_adapter as _baseline_adapter
from hi_policy import HierarchicalWorldModelPolicy, calibrate_latent_prior

os.environ["MUJOCO_GL"] = "egl"

# Backward-compatibility for torch.load on object checkpoints saved by hi_train:
# those pickles may reference classes under the dynamic module name
# `_baseline_lewm_module` (created by baseline_adapter). Touch one exported
# symbol so baseline_adapter registers that dynamic module in sys.modules
# before AutoCostModel unpickles.
_ = _baseline_adapter.ARPredictor

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}


def resolve_output_dir(cfg: DictConfig) -> Path:
    """Resolve directory used for videos and result text output.

    Uses policy parent directory by default and optionally appends output.subdir.
    """
    base_dir = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    output_subdir = str(cfg.output.get("subdir", "")).strip()
    if output_subdir:
        subdir = Path(output_subdir)
        if subdir.is_absolute() or ".." in subdir.parts:
            raise ValueError(
                "output.subdir must be a relative path without '..' segments."
            )
        base_dir = base_dir / subdir

    return base_dir


def list_video_inventory(output_dir: Path) -> dict[Path, int]:
    """Return mapping of video files to mtime_ns under output_dir."""
    inventory = {}
    if not output_dir.exists():
        return inventory

    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        try:
            inventory[path] = path.stat().st_mtime_ns
        except OSError:
            continue
    return inventory


def discover_new_video_files(
    output_dir: Path, before_inventory: dict[Path, int]
) -> list[Path]:
    """Find created/updated video files after evaluation."""
    after_inventory = list_video_inventory(output_dir)
    new_or_updated = []
    for path, mtime_ns in after_inventory.items():
        previous_mtime = before_inventory.get(path)
        if previous_mtime is None or mtime_ns > previous_mtime:
            new_or_updated.append(path)
    return sorted(
        set(new_or_updated),
        key=lambda path: (after_inventory.get(path, 0), str(path)),
    )


def extract_eval_index_from_video_name(path: Path, num_eval: int) -> int | None:
    """Best-effort parse of eval index encoded in a video file name."""
    stem = path.stem.lower()
    patterns = [
        r"(?:eval|episode|ep|rollout|traj|video)[_-]?(\d+)",
        r"[_-](\d+)$",
        r"(\d+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, stem):
            value = int(match.group(1))
            if 0 <= value < num_eval:
                return value
    return None


def map_eval_video_paths(video_files: list[Path], num_eval: int) -> list[str]:
    """Map eval index -> video path with filename-index hints and fallback order."""
    if num_eval <= 0:
        return []

    mapped = ["" for _ in range(num_eval)]
    remaining_files = []
    taken_indices = set()

    for path in video_files:
        idx = extract_eval_index_from_video_name(path, num_eval=num_eval)
        if idx is None or idx in taken_indices:
            remaining_files.append(path)
            continue
        mapped[idx] = str(path)
        taken_indices.add(idx)

    remaining_iter = iter(remaining_files)
    for eval_index in range(num_eval):
        if mapped[eval_index]:
            continue
        next_path = next(remaining_iter, None)
        if next_path is None:
            break
        mapped[eval_index] = str(next_path)

    return mapped


def extract_episode_successes(metrics: dict, expected_count: int) -> np.ndarray:
    """Read per-eval success vector and validate length."""
    episode_successes = np.asarray(metrics.get("episode_successes", []), dtype=bool)
    if episode_successes.shape[0] != expected_count:
        raise ValueError(
            "Mismatch between sampled evaluations and episode_successes: "
            f"{expected_count} samples vs {episode_successes.shape[0]} outcomes"
        )
    return episode_successes


def build_episode_outcomes(
    eval_episodes: np.ndarray,
    eval_start_idx: np.ndarray,
    episode_successes: np.ndarray,
    eval_video_paths: list[str],
) -> list[dict[str, str | int]]:
    """Build structured per-eval outcomes."""
    episode_ids = eval_episodes.tolist()
    start_steps = eval_start_idx.tolist()
    successes = episode_successes.tolist()
    if not (len(episode_ids) == len(start_steps) == len(successes)):
        raise ValueError(
            "Outcome inputs must have same length: "
            f"episode_ids={len(episode_ids)}, "
            f"start_steps={len(start_steps)}, "
            f"successes={len(successes)}"
        )

    outcomes = []
    for eval_index, (episode_id, start_step, success) in enumerate(
        zip(episode_ids, start_steps, successes)
    ):
        outcomes.append(
            {
                "eval_index": int(eval_index),
                "episode_id": int(episode_id),
                "start_step": int(start_step),
                "status": "PASS" if bool(success) else "FAIL",
                "video_path": eval_video_paths[eval_index]
                if eval_index < len(eval_video_paths)
                else "",
            }
        )
    return outcomes


def format_outcome_line(outcome: dict[str, str | int]) -> str:
    return (
        f"{outcome['status']}\t"
        f"eval_index={outcome['eval_index']}\t"
        f"episode_id={outcome['episode_id']}\t"
        f"start_step={outcome['start_step']}\t"
        f"video_path={outcome['video_path']}"
    )


def write_episode_manifest(manifest_path: Path, outcomes: list[dict[str, str | int]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        f.write("eval_index\tepisode_id\tstart_step\tstatus\tvideo_path\n")
        for outcome in outcomes:
            f.write(
                f"{outcome['eval_index']}\t"
                f"{outcome['episode_id']}\t"
                f"{outcome['start_step']}\t"
                f"{outcome['status']}\t"
                f"{outcome['video_path']}\n"
            )


def sample_eval_row_indices(valid_indices: np.ndarray, num_eval: int, seed: int) -> np.ndarray:
    """Sample unique dataset row indices used as evaluation starts.

    Args:
        valid_indices: 1D array of dataset rows eligible as start points.
        num_eval: Number of evaluation starts to sample.
        seed: RNG seed.

    Returns:
        Sorted 1D array of sampled dataset row indices.
    """
    valid_indices = np.asarray(valid_indices)
    if valid_indices.ndim != 1:
        raise ValueError("valid_indices must be a 1D array.")
    if num_eval <= 0:
        raise ValueError("num_eval must be > 0.")
    if len(valid_indices) < num_eval:
        raise ValueError(
            "Not enough valid starting points for evaluation: "
            f"requested {num_eval}, found {len(valid_indices)}."
        )

    g = np.random.default_rng(seed)
    sampled_positions = g.choice(len(valid_indices), size=num_eval, replace=False)
    return np.sort(valid_indices[sampled_positions])


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def build_process_map(cfg, dataset):
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def build_policy(cfg, model, dataset, process, transform):
    mode = str(cfg.planning.get("mode", "hierarchical")).lower()
    if mode == "flat":
        flat_cfg = swm.policy.PlanConfig(**cfg.plan_config)
        flat_solver = hydra.utils.instantiate(cfg.solver, model=model)
        return swm.policy.WorldModelPolicy(
            solver=flat_solver,
            config=flat_cfg,
            process=process,
            transform=transform,
        )

    if mode != "hierarchical":
        raise ValueError(
            f"Unsupported planning.mode='{mode}'. Use one of: hierarchical, flat."
        )

    high_cfg = swm.policy.PlanConfig(**cfg.planning.high.plan_config)
    low_cfg = swm.policy.PlanConfig(**cfg.planning.low.plan_config)
    high_solver = hydra.utils.instantiate(cfg.planning.high.solver, model=model)
    low_solver = hydra.utils.instantiate(cfg.planning.low.solver, model=model)

    high_bounds = None
    if bool(cfg.planning.high.latent_prior.get("enabled", True)):
        high_bounds = calibrate_latent_prior(
            model=model,
            dataset=dataset,
            cfg=cfg.planning.high.latent_prior,
            process=process,
            seed=int(cfg.seed),
        )
        print(
            "[hi_eval] calibrated high-level latent bounds "
            f"(chunks={int(high_bounds['num_chunks'])}, chunk_len={int(high_bounds['chunk_len'])})"
        )

    return HierarchicalWorldModelPolicy(
        model=model,
        high_solver=high_solver,
        low_solver=low_solver,
        high_config=high_cfg,
        low_config=low_cfg,
        macro_replan_interval=int(cfg.planning.high.replan_interval),
        process=process,
        transform=transform,
        high_latent_bounds=high_bounds,
    )


@hydra.main(version_base=None, config_path="./config/eval", config_name="hi_pusht")
def run(cfg: DictConfig):
    mode = str(cfg.planning.get("mode", "hierarchical")).lower()

    if mode == "hierarchical":
        high_plan_len = (
            int(cfg.planning.high.plan_config.horizon)
            * int(cfg.planning.high.plan_config.action_block)
        )
        low_plan_len = (
            int(cfg.planning.low.plan_config.horizon)
            * int(cfg.planning.low.plan_config.action_block)
        )
        assert high_plan_len <= int(cfg.eval.eval_budget), (
            "High-level plan length must be <= eval_budget"
        )
        assert low_plan_len <= int(cfg.eval.eval_budget), (
            "Low-level plan length must be <= eval_budget"
        )
    else:
        assert (
            cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
        ), "Planning horizon must be smaller than or equal to eval_budget"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)
    process = build_process_map(cfg, stats_dataset)

    policy_name = cfg.get("policy", "random")
    if policy_name != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        policy = build_policy(cfg, model, dataset, process, transform)
    else:
        policy = swm.policy.RandomPolicy()

    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_inventory_before = list_video_inventory(output_dir)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    sampled_indices = sample_eval_row_indices(
        valid_indices=valid_indices,
        num_eval=int(cfg.eval.num_eval),
        seed=int(cfg.seed),
    )
    eval_episodes = dataset.get_row_data(sampled_indices)[col_name]
    eval_start_idx = dataset.get_row_data(sampled_indices)["step_idx"]

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=output_dir,
    )
    end_time = time.time()
    video_files = discover_new_video_files(output_dir, before_inventory=video_inventory_before)

    print(metrics)
    episode_successes = extract_episode_successes(
        metrics=metrics,
        expected_count=len(eval_episodes),
    )
    eval_video_paths = map_eval_video_paths(video_files=video_files, num_eval=len(eval_episodes))
    outcomes = build_episode_outcomes(
        eval_episodes=eval_episodes,
        eval_start_idx=eval_start_idx,
        episode_successes=episode_successes,
        eval_video_paths=eval_video_paths,
    )
    failed_outcomes = [o for o in outcomes if o["status"] == "FAIL"]
    passed_outcomes = [o for o in outcomes if o["status"] == "PASS"]

    print("==== EPISODE OUTCOMES ====")
    for outcome in outcomes:
        print(format_outcome_line(outcome))
    print("==== FAILED EPISODES ====")
    for outcome in failed_outcomes:
        print(format_outcome_line(outcome))
    print("==== PASSED EPISODES ====")
    for outcome in passed_outcomes:
        print(format_outcome_line(outcome))

    results_path = output_dir / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = results_path.with_name(f"{results_path.stem}_episodes.tsv")

    with results_path.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")
        f.write("==== EPISODE OUTCOMES ====\n")
        for outcome in outcomes:
            f.write(f"{format_outcome_line(outcome)}\n")

    write_episode_manifest(manifest_path=manifest_path, outcomes=outcomes)
    print(f"Saved episode manifest to {manifest_path}")


if __name__ == "__main__":
    run()
