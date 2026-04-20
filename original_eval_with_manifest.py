import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import time

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from eval_dataset_metrics import evaluate_from_dataset_with_optional_metrics


# LeWM checkpoints were serialized with classes from a top-level `jepa` module.
# Add the vendored source directory so torch.load can resolve that module on clusters
# where the package is not installed separately.
_VENDORED_LEWM_DIR = Path(__file__).resolve().parent / "third_party" / "lewm"
if _VENDORED_LEWM_DIR.is_dir():
    sys.path.insert(0, str(_VENDORED_LEWM_DIR))


def resolve_output_dir(cfg: DictConfig) -> Path:
    """Resolve directory used for videos and result outputs."""
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


def format_episode_outcomes(eval_episodes, eval_start_idx, episode_successes):
    lines = []
    for eval_index, (episode_id, start_step, success) in enumerate(
        zip(eval_episodes.tolist(), eval_start_idx.tolist(), episode_successes.tolist())
    ):
        status = "PASS" if success else "FAIL"
        lines.append(
            f"{status}\teval_index={eval_index}\tepisode_id={episode_id}\tstart_step={start_step}"
        )
    return lines


def should_compute_pusht_block_only_metric(cfg: DictConfig) -> bool:
    return str(cfg.world.env_name).strip() == "swm/PushT-v1"


@hydra.main(
    version_base=None,
    config_path="third_party/lewm/config/eval",
    config_name="pusht",
)
def run(cfg: DictConfig):
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

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    policy = cfg.get("policy", "random")
    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    output_root = resolve_output_dir(cfg)
    output_root.mkdir(parents=True, exist_ok=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    print(random_episode_indices)

    eval_rows = dataset.get_row_data(random_episode_indices)
    eval_episodes = eval_rows[col_name]
    eval_start_idx = eval_rows["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

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
        video_path=output_root,
    )
    end_time = time.time()

    print(metrics)
    if "success_rate_block_only" in metrics:
        print(f"block_only_success_rate: {metrics['success_rate_block_only']}")

    episode_successes = np.asarray(metrics.get("episode_successes", []), dtype=bool)
    if episode_successes.shape[0] != len(eval_episodes):
        raise ValueError(
            "Mismatch between sampled evaluations and episode_successes: "
            f"{len(eval_episodes)} samples vs {episode_successes.shape[0]} outcomes"
        )

    episode_successes_block_only = np.asarray(
        metrics.get("episode_successes_block_only", []),
        dtype=bool,
    )
    has_block_only = "episode_successes_block_only" in metrics
    if has_block_only and episode_successes_block_only.shape[0] != len(eval_episodes):
        raise ValueError(
            "Mismatch between sampled evaluations and episode_successes_block_only: "
            f"{len(eval_episodes)} samples vs {episode_successes_block_only.shape[0]} outcomes"
        )

    outcome_lines = format_episode_outcomes(
        eval_episodes=eval_episodes,
        eval_start_idx=eval_start_idx,
        episode_successes=episode_successes,
    )
    failed_lines = [line for line in outcome_lines if line.startswith("FAIL")]
    passed_lines = [line for line in outcome_lines if line.startswith("PASS")]

    print("==== EPISODE OUTCOMES ====")
    for line in outcome_lines:
        print(line)
    print("==== FAILED EPISODES ====")
    for line in failed_lines:
        print(line)
    print("==== PASSED EPISODES ====")
    for line in passed_lines:
        print(line)
    if has_block_only:
        block_only_lines = format_episode_outcomes(
            eval_episodes=eval_episodes,
            eval_start_idx=eval_start_idx,
            episode_successes=episode_successes_block_only,
        )
        block_only_failed = [line for line in block_only_lines if line.startswith("FAIL")]
        block_only_passed = [line for line in block_only_lines if line.startswith("PASS")]
        print("==== BLOCK-ONLY EPISODE OUTCOMES ====")
        for line in block_only_lines:
            print(line)
        print("==== BLOCK-ONLY FAILED EPISODES ====")
        for line in block_only_failed:
            print(line)
        print("==== BLOCK-ONLY PASSED EPISODES ====")
        for line in block_only_passed:
            print(line)

    results_path = output_root / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = results_path.with_name(f"{results_path.stem}_episodes.tsv")

    with results_path.open("w") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        if "success_rate_block_only" in metrics:
            f.write(f"block_only_success_rate: {metrics['success_rate_block_only']}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")
        f.write("==== EPISODE OUTCOMES ====\n")
        for line in outcome_lines:
            f.write(f"{line}\n")
        if has_block_only:
            f.write("==== BLOCK-ONLY EPISODE OUTCOMES ====\n")
            for line in block_only_lines:
                f.write(f"{line}\n")

    with manifest_path.open("w") as f:
        header = "eval_index\tepisode_id\tstart_step\tstatus"
        if has_block_only:
            header += "\tstatus_block_only"
        f.write(f"{header}\n")
        for row in zip(
            range(len(eval_episodes)),
            eval_episodes.tolist(),
            eval_start_idx.tolist(),
            episode_successes.tolist(),
            episode_successes_block_only.tolist() if has_block_only else [""] * len(eval_episodes),
        ):
            eval_index, episode_id, start_step, success, block_success = row
            status = "PASS" if success else "FAIL"
            line = f"{eval_index}\t{episode_id}\t{start_step}\t{status}"
            if has_block_only:
                block_status = "PASS" if block_success else "FAIL"
                line += f"\t{block_status}"
            f.write(f"{line}\n")

    print(f"Saved episode manifest to {manifest_path}")


if __name__ == "__main__":
    run()
