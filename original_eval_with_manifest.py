import os
import sys
from pathlib import Path

_MUJOCO_GL_WAS_PRESET = "MUJOCO_GL" in os.environ
os.environ.setdefault("MUJOCO_GL", "egl")

import time

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


# LeWM checkpoints were serialized with classes from a top-level `jepa` module.
# Add the vendored source directory so torch.load can resolve that module on clusters
# where the package is not installed separately.
_VENDORED_LEWM_DIR = Path(__file__).resolve().parent / "third_party" / "lewm"
if _VENDORED_LEWM_DIR.is_dir():
    sys.path.insert(0, str(_VENDORED_LEWM_DIR))


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


def resolve_runtime_device(cfg: DictConfig) -> str:
    """Resolve requested runtime device with CUDA availability fallback."""
    raw_value = cfg.get("device", os.environ.get("EVAL_DEVICE", "auto"))
    requested = str(raw_value).strip().lower()
    if not requested:
        requested = "auto"

    normalized = "cuda" if requested.startswith("cuda") else requested
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "Unsupported runtime device "
            f"'{requested}'. Supported values: auto, cpu, cuda."
        )

    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        print(
            "[original_eval] requested device='cuda' but CUDA is unavailable; "
            "falling back to CPU."
        )
        return "cpu"
    return requested


@hydra.main(
    version_base=None,
    config_path="third_party/lewm/config/eval",
    config_name="pusht",
)
def run(cfg: DictConfig):
    runtime_device = resolve_runtime_device(cfg)
    if "solver" in cfg and "device" in cfg.solver:
        cfg.solver.device = runtime_device
    if runtime_device == "cpu" and not _MUJOCO_GL_WAS_PRESET:
        os.environ["MUJOCO_GL"] = "osmesa"
    print(
        f"[original_eval] runtime_device={runtime_device}, "
        f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}"
    )

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
        model = model.to(runtime_device)
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

    output_root = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

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
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=output_root,
    )
    end_time = time.time()

    print(metrics)

    episode_successes = np.asarray(metrics.get("episode_successes", []), dtype=bool)
    if episode_successes.shape[0] != len(eval_episodes):
        raise ValueError(
            "Mismatch between sampled evaluations and episode_successes: "
            f"{len(eval_episodes)} samples vs {episode_successes.shape[0]} outcomes"
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

    results_path = output_root / cfg.output.filename
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
        for line in outcome_lines:
            f.write(f"{line}\n")

    with manifest_path.open("a") as f:
        f.write("eval_index\tepisode_id\tstart_step\tstatus\n")
        for eval_index, episode_id, start_step, success in zip(
            range(len(eval_episodes)),
            eval_episodes.tolist(),
            eval_start_idx.tolist(),
            episode_successes.tolist(),
        ):
            status = "PASS" if success else "FAIL"
            f.write(
                f"{eval_index}\t{episode_id}\t{start_step}\t{status}\n"
            )

    print(f"Saved episode manifest to {manifest_path}")


if __name__ == "__main__":
    run()
