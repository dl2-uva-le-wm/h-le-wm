"""Codebook rollout experiment.

For each entry in the VQ codebook, runs one environment rollout by replacing
the high-level CEM with a fixed codebook vector. The low-level CEM is kept
intact so the agent actually moves in the environment.

Each video is written with a stable action-index filename:
    videos/action_000.mp4, videos/action_001.mp4, ...

Usage:
    python scripts/codebook_rollout_experiment.py \\
        --config-name=codebook_rollout \\
        policy=runs/<run_name>_epoch_<N> \\
        experiment.n_frames=50 \\
        experiment.num_actions=100 \\
        experiment.output_dir=/path/to/output

Submit on Snellius via:
    sbatch --export=ALL,CHECKPOINT_NAME=<run>_epoch_<N> \\
           jobs/diagnostics/codebook_rollout.sh
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

# Needed for unpickling checkpoints that reference baseline adapter classes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "lewm"))
import baseline_adapter as _baseline_adapter  # noqa: F401  # registers dynamic module

_ = _baseline_adapter.ARPredictor

from hi_policy import HierarchicalWorldModelPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (inlined from hi_eval to avoid its module-level side-effects)
# ---------------------------------------------------------------------------

def _img_transform(cfg: DictConfig):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def _get_dataset(cfg: DictConfig, dataset_name: str):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def _build_process_map(cfg: DictConfig, dataset) -> dict:
    process: dict = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        scaler = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        scaler.fit(col_data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return process


def _sample_eval_row_indices(
    valid_indices: np.ndarray, num_eval: int, seed: int
) -> np.ndarray:
    g = np.random.default_rng(seed)
    positions = g.choice(len(valid_indices), size=num_eval, replace=False)
    return np.sort(valid_indices[positions])


# ---------------------------------------------------------------------------
# Codebook utilities
# ---------------------------------------------------------------------------

def get_codebook(model: torch.nn.Module) -> torch.Tensor | None:
    """Return (num_codes, latent_dim) codebook weight tensor, or None."""
    enc = getattr(model, "latent_action_encoder", None)
    vq = getattr(enc, "quantizer", None)
    cb = getattr(vq, "codebook", None)
    return cb.weight if cb is not None else None


def _resolve_num_actions(cfg: DictConfig, total_codes: int) -> int:
    """Resolve how many codebook actions to render."""
    num_actions_cfg = cfg.experiment.get("num_actions", None)
    num_codes_cfg = cfg.experiment.get("num_codes", None)

    if num_actions_cfg is not None and num_codes_cfg is not None:
        if int(num_actions_cfg) != int(num_codes_cfg):
            raise ValueError(
                "experiment.num_actions and deprecated experiment.num_codes "
                "were both set to different values."
            )

    requested = num_actions_cfg if num_actions_cfg is not None else num_codes_cfg
    if requested is None:
        return total_codes

    num_actions = int(requested)
    if num_actions <= 0:
        raise ValueError(f"experiment.num_actions must be positive; got {num_actions}.")
    if num_actions > total_codes:
        raise ValueError(
            f"Requested {num_actions} actions, but the codebook has only {total_codes} entries."
        )
    return num_actions


# ---------------------------------------------------------------------------
# Policy: fixed codebook entry replaces high-level CEM
# ---------------------------------------------------------------------------

class CodebookFixedActionPolicy(HierarchicalWorldModelPolicy):
    """High-level CEM replaced by a fixed VQ codebook lookup.

    At each high-level replanning step the policy looks up `codebook[idx]`
    and uses it as the macro-action to compute a subgoal via `rollout_high`.
    The low-level CEM is unchanged.
    """

    def __init__(
        self,
        *,
        codebook: torch.Tensor,
        model: torch.nn.Module,
        low_solver: Any,
        low_config: Any,
        macro_replan_interval: int = 5,
        process: dict | None = None,
        transform: dict | None = None,
        high_latent_bounds: dict | None = None,
    ) -> None:
        class _DummyHighConfig:
            horizon = 1
            receding_horizon = 1
            action_block = 1
            warm_start = False

        super().__init__(
            model=model,
            high_solver=None,   # unused — we override _plan_high and set_env
            low_solver=low_solver,
            high_config=_DummyHighConfig(),
            low_config=low_config,
            macro_replan_interval=macro_replan_interval,
            process=process,
            transform=transform,
            high_latent_bounds=high_latent_bounds,
        )
        self._codebook = codebook   # (num_codes, latent_dim)
        self._codebook_idx: int = 0

    def set_codebook_index(self, idx: int) -> None:
        """Switch to codebook entry `idx` and reset episode state."""
        self._codebook_idx = idx
        self._z_subgoal = None
        self._action_buffer.clear()
        self._steps_since_high = 10 ** 9
        self._next_low_init = None
        self._next_high_init = None

    def set_env(self, env: Any) -> None:
        """Attach environment; configure only the low-level solver."""
        self.env = env
        n_envs = int(getattr(env, "num_envs", 1))

        self.low_solver.configure(
            action_space=env.action_space,
            n_envs=n_envs,
            config=self.low_cfg,
        )
        self._low_grouped_action_dim = int(self.low_solver.action_dim)
        # high_solver is None — no configure() call.

        self._action_buffer = deque(maxlen=self.flatten_receding_horizon_low)
        self._next_low_init = None
        self._next_high_init = None
        self._z_subgoal = None
        self._steps_since_high = 10 ** 9

    @torch.inference_mode()
    def _plan_high(self, *, z_init: torch.Tensor, z_goal: torch.Tensor) -> None:
        """Use fixed codebook entry instead of CEM to compute subgoal."""
        n_envs = z_init.size(0)
        device = z_init.device

        codebook_vec = self._codebook[self._codebook_idx].to(device)  # (latent_dim,)
        # rollout_high expects (B, T, D_l) — batch, time=1, latent_dim
        macro_seq = codebook_vec.reshape(1, 1, -1).expand(n_envs, 1, -1).contiguous()

        # rollout_high: (B, T, D_l) → (B, S=1, T=1, D) because T=1
        pred = self.model.rollout_high(z_init, macro_seq)
        self._z_subgoal = pred[:, 0, 0, :]   # (B, D)
        self._steps_since_high = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _select_start_step(
    dataset,
    col_name: str,
    goal_offset_steps: int,
    seed: int,
) -> tuple[int, int]:
    """Return (dataset_row_index, episode_id) for a valid start state."""
    ep_indices_all = dataset.get_col_data(col_name)
    step_indices_all = dataset.get_col_data("step_idx")

    unique_eps = np.unique(ep_indices_all)
    max_start_per_ep: dict[int, int] = {}
    for ep_id in unique_eps:
        mask = ep_indices_all == ep_id
        ep_len = int(step_indices_all[mask].max()) + 1
        max_start_per_ep[ep_id] = ep_len - goal_offset_steps - 1

    max_start_per_row = np.array([max_start_per_ep[ep] for ep in ep_indices_all])
    valid_mask = step_indices_all <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    if len(valid_indices) == 0:
        raise RuntimeError("No valid starting rows found in dataset.")

    sampled = _sample_eval_row_indices(valid_indices, num_eval=1, seed=seed)
    row_idx = int(sampled[0])
    episode_id = int(ep_indices_all[row_idx])
    return row_idx, episode_id


@hydra.main(version_base=None, config_path="../config/eval", config_name="codebook_rollout")
def run(cfg: DictConfig) -> None:
    # Snap n_frames to a multiple of low action_block so macro-steps are complete.
    low_action_block = int(cfg.planning.low.plan_config.action_block)
    n_frames = int(cfg.experiment.n_frames)
    n_frames_snapped = max(low_action_block, (n_frames // low_action_block) * low_action_block)
    if n_frames_snapped != n_frames:
        print(f"[codebook_rollout] n_frames snapped {n_frames} → {n_frames_snapped} "
              f"(multiple of low.action_block={low_action_block})")
    n_frames = n_frames_snapped

    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[codebook_rollout] output_dir={output_dir}")

    # ----- model -----
    model_device = str(cfg.planning.low.solver.device)
    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to(model_device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    # ----- codebook -----
    codebook = get_codebook(model)
    if codebook is None:
        raise RuntimeError(
            "No VQ codebook found in model.latent_action_encoder.quantizer. "
            "This experiment requires a checkpoint trained with a VQ action encoder."
        )
    total_codes = int(codebook.size(0))
    num_actions = _resolve_num_actions(cfg, total_codes)
    index_width = max(3, len(str(total_codes - 1)))
    videos_dir = output_dir / "videos"
    work_dir = output_dir / "per_action_work"
    videos_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[codebook_rollout] codebook size={total_codes}, running {num_actions} actions, "
          f"latent_dim={codebook.size(1)}")

    # ----- dataset / preprocessing -----
    dataset = _get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    process = _build_process_map(cfg, dataset)
    transform = {"pixels": _img_transform(cfg), "goal": _img_transform(cfg)}

    # Single starting state reused for all codebook entries.
    start_row_idx, start_episode_id = _select_start_step(
        dataset=dataset,
        col_name=col_name,
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        seed=int(cfg.experiment.start_step_seed),
    )
    start_step = int(dataset.get_row_data([start_row_idx])["step_idx"][0])
    print(f"[codebook_rollout] start episode={start_episode_id}, step={start_step}")

    # ----- low-level CEM solver -----
    low_cfg = swm.policy.PlanConfig(**cfg.planning.low.plan_config)
    low_solver = hydra.utils.instantiate(cfg.planning.low.solver, model=model)

    # ----- policy -----
    policy = CodebookFixedActionPolicy(
        codebook=codebook.detach(),
        model=model,
        low_solver=low_solver,
        low_config=low_cfg,
        macro_replan_interval=int(cfg.planning.high.replan_interval),
        process=process,
        transform=transform,
    )

    # ----- world (num_envs=1, one episode at a time) -----
    cfg.world.max_episode_steps = 2 * n_frames
    world = swm.World(**cfg.world, image_shape=(224, 224))
    world.set_policy(policy)

    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)

    # ----- main loop -----
    summary: list[dict] = []
    t_start_all = time.time()

    for idx in range(num_actions):
        t0 = time.time()
        action_id = f"action_{idx:0{index_width}d}"
        print(
            f"[codebook_rollout] {action_id} ({idx + 1}/{num_actions}) ...",
            end=" ",
            flush=True,
        )

        policy.set_codebook_index(idx)

        entry_video_dir = work_dir / action_id
        entry_video_dir.mkdir(parents=True, exist_ok=True)

        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=[start_step],
            goal_offset_steps=int(cfg.eval.goal_offset_steps),
            eval_budget=n_frames,
            episodes_idx=[start_episode_id],
            callables=callables,
            video_path=entry_video_dir,
        )

        elapsed = time.time() - t0
        successes = list(metrics.get("episode_successes", [False]))
        success = bool(successes[0]) if successes else False

        video_files = sorted(entry_video_dir.rglob("*.mp4"))
        if len(video_files) != 1:
            raise RuntimeError(
                f"Expected exactly one video for action index {idx}, "
                f"found {len(video_files)} in {entry_video_dir}."
            )

        canonical_video_path = videos_dir / f"{action_id}.mp4"
        if canonical_video_path.exists():
            canonical_video_path.unlink()
        video_files[0].replace(canonical_video_path)
        video_path_str = str(canonical_video_path)

        cb_vec = codebook[idx].detach().cpu()
        entry_summary = {
            "action_index": idx,
            "action_id": action_id,
            "video_filename": canonical_video_path.name,
            "codebook_index": idx,
            "codebook_vector_norm": float(cb_vec.norm().item()),
            "success": success,
            "elapsed_s": round(elapsed, 2),
            "video_path": video_path_str,
            "metrics": {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in metrics.items()
            },
        }
        summary.append(entry_summary)

        status = "SUCCESS" if success else "fail"
        print(f"[{status}] ({elapsed:.1f}s)  video={video_path_str or 'none'}")

    total_elapsed = time.time() - t_start_all
    print(f"\n[codebook_rollout] Done. {num_actions} rollouts in {total_elapsed:.1f}s "
          f"({total_elapsed / num_actions:.1f}s/action average)")

    # ----- save summary -----
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "policy": cfg.policy,
                "n_frames": n_frames,
                "num_actions_total": total_codes,
                "num_actions_run": num_actions,
                "num_codes_total": total_codes,
                "num_codes_run": num_actions,
                "video_dir": str(videos_dir),
                "video_naming": "videos/action_<zero-padded action index>.mp4",
                "start_episode_id": start_episode_id,
                "start_step": start_step,
                "total_elapsed_s": round(total_elapsed, 2),
                "entries": summary,
            },
            f,
            indent=2,
        )
    print(f"[codebook_rollout] Summary saved to {summary_path}")

    cfg_path = output_dir / "config.yaml"
    with cfg_path.open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    run()
