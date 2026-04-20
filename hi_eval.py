import os
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import baseline_adapter as _baseline_adapter
from eval_dataset_metrics import evaluate_from_dataset_with_optional_metrics
from hi_policy import HierarchicalWorldModelPolicy, calibrate_latent_prior

os.environ["MUJOCO_GL"] = "egl"

# Backward-compatibility for torch.load on object checkpoints saved by hi_train:
# those pickles may reference classes under the dynamic module name
# `_baseline_lewm_module` (created by baseline_adapter). Touch one exported
# symbol so baseline_adapter registers that dynamic module in sys.modules
# before AutoCostModel unpickles.
_ = _baseline_adapter.ARPredictor


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


def should_compute_pusht_block_only_metric(cfg: DictConfig) -> bool:
    return str(cfg.world.env_name).strip() == "swm/PushT-v1"


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
    metrics = evaluate_from_dataset_with_optional_metrics(
        world=world,
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        enable_pusht_block_only=should_compute_pusht_block_only_metric(cfg),
        video_path=output_dir,
    )
    end_time = time.time()

    print(metrics)
    if "success_rate_block_only" in metrics:
        print(f"block_only_success_rate: {metrics['success_rate_block_only']}")
    results_path = output_dir / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        if "success_rate_block_only" in metrics:
            f.write(f"block_only_success_rate: {metrics['success_rate_block_only']}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
