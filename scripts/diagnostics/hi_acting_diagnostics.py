#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import stable_worldmodel as swm
import torch
from gymnasium.spaces import Box
from omegaconf import OmegaConf
from sklearn import preprocessing

from hi_diagnostics import (
    ReferenceStats,
    append_tsv,
    base_result as _unused_base_result,  # imported to keep helper source colocated
    build_action_scaler,
    build_reference_stats,
    contiguous_valid_starts,
    encode_macro_actions,
    encode_pixels_last,
    finalize_result as _unused_finalize_result,  # imported to keep helper source colocated
    get_episode_col_name,
    goal_tokens,
    img_transform,
    infer_latent_dim,
    infer_low_action_input_dim,
    infer_macro_input_dim,
    mahalanobis_sq,
    parse_policy_run_info,
    partition_total,
    preprocess_actions,
    resolve_cache_dir,
    resolve_device,
    sample_indices,
    stats_percentiles,
    to_serializable,
    write_json,
)
from h_le_wm.eval.hierarchical import build_process_map
from h_le_wm.planning.policies import (
    HierarchicalWorldModelPolicy,
    StagedHierarchicalWorldModelPolicy,
    calibrate_latent_prior,
)


@dataclass(slots=True)
class ActingDiagnosticConfig:
    policy: str
    experiment_kind: str
    dataset_name: str = "pusht_expert_train"
    eval_config: str = "h_le_wm/config/eval/hi_pusht.yaml"
    cache_dir: str | None = None
    img_size: int = 224
    num_eval: int = 50
    goal_offset_steps: int = 50
    eval_budget: int = 50
    high_horizon: int = 2
    low_horizon: int = 2
    low_receding_horizon: int = 1
    high_num_samples: int = 1500
    high_iters: int = 40
    high_topk: int = 10
    low_num_samples: int = 900
    low_iters: int = 20
    low_topk: int = 150
    frame_skip: int = 5
    seed: int = 42
    device: str = "auto"
    subgoal_offsets: tuple[int, ...] = (2, 3, 5)
    num_reference_samples: int = 4096
    save_json: str | None = None
    save_npz: str | None = None
    append_tsv: str | None = None


@dataclass(slots=True)
class ActingContext:
    cfg: ActingDiagnosticConfig
    eval_cfg: Any
    device: torch.device
    rng: np.random.Generator
    dataset: Any
    world: swm.World
    model: torch.nn.Module
    process: dict[str, Any]
    transform: dict[str, Any]
    action_scaler: preprocessing.StandardScaler
    episode_ids: np.ndarray
    step_idx: np.ndarray | None
    action: np.ndarray
    latent_dim: int
    raw_action_dim: int
    macro_input_dim: int
    group: int
    low_action_input_dim: int
    action_ref: ReferenceStats
    macro_ref_cache: dict[int, ReferenceStats]


@dataclass(slots=True)
class PreparedBatch:
    sampled_indices: np.ndarray
    episodes_idx: np.ndarray
    start_steps: np.ndarray
    data: list[dict[str, Any]]
    init_step_np: dict[str, np.ndarray]
    goal_step_np: dict[str, np.ndarray]
    init_step_broadcast: dict[str, np.ndarray]
    goal_step_broadcast: dict[str, np.ndarray]
    future_pixels_bthwc: np.ndarray
    future_states: np.ndarray | None
    future_latents: torch.Tensor
    goal_latent: torch.Tensor
    start_latent: torch.Tensor


def maybe_save_npz(path: str | None, arrays: dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)


def base_result(ctx: ActingContext) -> dict[str, Any]:
    run_name, checkpoint_epoch = parse_policy_run_info(ctx.cfg.policy)
    return {
        "run_name": run_name,
        "checkpoint_epoch": checkpoint_epoch,
        "policy": ctx.cfg.policy,
        "dataset_name": ctx.cfg.dataset_name,
        "experiment_kind": ctx.cfg.experiment_kind,
        "device": str(ctx.device),
        "seed": int(ctx.cfg.seed),
        "goal_offset_steps": int(ctx.cfg.goal_offset_steps),
        "eval_budget": int(ctx.cfg.eval_budget),
        "num_eval": int(ctx.cfg.num_eval),
        "goal_tokens": int(goal_tokens(ctx.cfg)),
        "high_horizon": int(ctx.cfg.high_horizon),
        "low_horizon": int(ctx.cfg.low_horizon),
        "low_receding_horizon": int(ctx.cfg.low_receding_horizon),
        "latent_action_dim": int(ctx.latent_dim),
        "macro_input_dim": int(ctx.macro_input_dim),
        "low_action_input_dim": int(ctx.low_action_input_dim),
        "raw_action_dim": int(ctx.raw_action_dim),
        "group_factor": int(ctx.group),
        "result_status": "ok",
    }


def finalize_result(
    ctx: ActingContext,
    result: dict[str, Any],
    *,
    tsv_columns: Sequence[str],
    tsv_row: dict[str, Any],
    npz_arrays: dict[str, Any] | None = None,
) -> dict[str, Any]:
    write_json(ctx.cfg.save_json, result)
    append_tsv(ctx.cfg.append_tsv, tsv_columns, tsv_row)
    if npz_arrays:
        maybe_save_npz(ctx.cfg.save_npz, npz_arrays)
    return result


def get_episodes_length(dataset, episodes):
    col_name = get_episode_col_name(dataset)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.asarray(lengths)


def sample_eval_row_indices(valid_indices: np.ndarray, num_eval: int, seed: int) -> np.ndarray:
    valid_indices = np.asarray(valid_indices)
    if valid_indices.ndim != 1:
        raise ValueError("valid_indices must be 1D.")
    if len(valid_indices) < num_eval:
        raise ValueError(f"Not enough valid starting points: need {num_eval}, got {len(valid_indices)}.")
    g = np.random.default_rng(seed)
    picked = g.choice(len(valid_indices), size=num_eval, replace=False)
    return np.sort(valid_indices[picked])


def encode_pixels_sequence(
    model: torch.nn.Module,
    pixels_bthwc: np.ndarray,
    tfm,
    device: torch.device,
    *,
    batch_size: int = 128,
) -> torch.Tensor:
    b, t = pixels_bthwc.shape[:2]
    flat = pixels_bthwc.reshape(b * t, *pixels_bthwc.shape[2:])
    encoded_chunks = []
    for start in range(0, flat.shape[0], batch_size):
        encoded_chunks.append(encode_pixels_last(model, flat[start : start + batch_size], tfm, device))
    encoded = torch.cat(encoded_chunks, dim=0)
    return encoded.reshape(b, t, -1)


def build_action_reference(ctx: ActingContext) -> ReferenceStats:
    starts = sample_valid_starts_for_raw_span(ctx, raw_span=ctx.group, n=ctx.cfg.num_reference_samples)
    chunks_raw = np.stack([ctx.action[s : s + ctx.group] for s in starts], axis=0)
    chunks_tok = preprocess_actions(
        chunks_raw,
        ctx.action_scaler,
        chunk_len_tokens=1,
        group=ctx.group,
    ).reshape(len(starts), ctx.low_action_input_dim)
    token_t = torch.from_numpy(chunks_tok.astype(np.float32)).to(ctx.device)
    return build_reference_stats(token_t, span_tokens=1)


def sample_valid_starts_for_raw_span(ctx: ActingContext, *, raw_span: int, n: int) -> np.ndarray:
    valid_starts = contiguous_valid_starts(
        episode_ids=ctx.episode_ids,
        step_idx=ctx.step_idx,
        seq_len=int(ctx.action.shape[0]),
        chunk_len=raw_span + 1,
    )
    if valid_starts.size == 0:
        raise ValueError(f"No valid contiguous starts found for raw_span={raw_span}.")
    return sample_indices(ctx.rng, valid_starts, n)


def build_macro_reference(ctx: ActingContext, span_tokens: int) -> ReferenceStats:
    span_tokens = int(span_tokens)
    if span_tokens in ctx.macro_ref_cache:
        return ctx.macro_ref_cache[span_tokens]
    raw_span = span_tokens * ctx.group
    starts = sample_valid_starts_for_raw_span(ctx, raw_span=raw_span, n=ctx.cfg.num_reference_samples)
    chunks_raw = np.stack([ctx.action[s : s + raw_span] for s in starts], axis=0)
    chunks_tok = preprocess_actions(
        chunks_raw,
        ctx.action_scaler,
        chunk_len_tokens=span_tokens,
        group=ctx.group,
    )
    samples = encode_macro_actions(ctx.model, chunks_tok, device=ctx.device)
    ref = build_reference_stats(samples, span_tokens=span_tokens)
    ctx.macro_ref_cache[span_tokens] = ref
    return ref


def build_context(cfg: ActingDiagnosticConfig) -> ActingContext:
    device = resolve_device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    cache_dir = resolve_cache_dir(cfg.cache_dir)
    eval_cfg = OmegaConf.load(cfg.eval_config)
    eval_cfg.cache_dir = str(cache_dir)
    eval_cfg.eval.dataset_name = cfg.dataset_name
    eval_cfg.eval.img_size = cfg.img_size
    eval_cfg.world.num_envs = cfg.num_eval
    eval_cfg.world.max_episode_steps = 2 * int(cfg.eval_budget)

    dataset = swm.data.HDF5Dataset(
        cfg.dataset_name,
        keys_to_cache=eval_cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )
    process = build_process_map(eval_cfg, dataset)
    transform = {
        "pixels": img_transform(cfg.img_size),
        "goal": img_transform(cfg.img_size),
    }

    model = swm.policy.AutoCostModel(cfg.policy, cache_dir=cache_dir).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    world = swm.World(**eval_cfg.world, image_shape=(224, 224))

    episode_col = get_episode_col_name(dataset)
    episode_ids = np.asarray(dataset.get_col_data(episode_col))
    step_idx = np.asarray(dataset.get_col_data("step_idx")) if "step_idx" in dataset.column_names else None
    action = np.asarray(dataset.get_col_data("action"))
    raw_action_dim = int(action.shape[1])
    latent_dim = infer_latent_dim(model)
    macro_input_dim = infer_macro_input_dim(model)
    if macro_input_dim is None:
        macro_input_dim = raw_action_dim
    if macro_input_dim % raw_action_dim != 0:
        raise ValueError(
            f"Model macro input dim {macro_input_dim} is not divisible by dataset action dim {raw_action_dim}."
        )
    group = macro_input_dim // raw_action_dim
    low_action_input_dim = infer_low_action_input_dim(model)
    action_scaler = build_action_scaler(dataset)

    tmp = ActingContext(
        cfg=cfg,
        eval_cfg=eval_cfg,
        device=device,
        rng=rng,
        dataset=dataset,
        world=world,
        model=model,
        process=process,
        transform=transform,
        action_scaler=action_scaler,
        episode_ids=episode_ids,
        step_idx=step_idx,
        action=action,
        latent_dim=latent_dim,
        raw_action_dim=raw_action_dim,
        macro_input_dim=macro_input_dim,
        group=group,
        low_action_input_dim=low_action_input_dim,
        action_ref=None,  # type: ignore[arg-type]
        macro_ref_cache={},
    )
    tmp.action_ref = build_action_reference(tmp)
    return tmp


def prepare_eval_batch(ctx: ActingContext) -> PreparedBatch:
    dataset = ctx.dataset
    col_name = get_episode_col_name(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - ctx.cfg.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    sampled_indices = sample_eval_row_indices(valid_indices, num_eval=int(ctx.cfg.num_eval), seed=int(ctx.cfg.seed))
    eval_rows = dataset.get_row_data(sampled_indices)
    eval_episodes = np.asarray(eval_rows[col_name])
    eval_start_idx = np.asarray(eval_rows["step_idx"])
    end_steps = eval_start_idx + int(ctx.cfg.goal_offset_steps)
    data = dataset.load_chunk(eval_episodes, eval_start_idx, end_steps)
    columns = dataset.column_names

    init_step_per_env: dict[str, list[Any]] = {}
    goal_step_per_env: dict[str, list[Any]] = {}
    for col in columns:
        init_step_per_env[col] = []
        goal_step_per_env[col] = []

    for ep in data:
        for col in columns:
            if col.startswith("goal"):
                continue
            val = ep[col]
            if col.startswith("pixels"):
                val = val.permute(0, 2, 3, 1)
            if not isinstance(val, (torch.Tensor, np.ndarray)):
                continue
            init_data = val[0]
            goal_data = val[-1]
            if isinstance(init_data, torch.Tensor):
                init_data = init_data.numpy()
            if isinstance(goal_data, torch.Tensor):
                goal_data = goal_data.numpy()
            init_step_per_env[col].append(init_data)
            goal_step_per_env[col].append(goal_data)

    init_step_np = {k: np.stack(v) for k, v in init_step_per_env.items() if len(v) > 0}
    goal_step_np = {}
    for key, value in goal_step_per_env.items():
        if not value:
            continue
        out_key = "goal" if key == "pixels" else f"goal_{key}"
        goal_step_np[out_key] = np.stack(value)

    future_pixels = []
    future_states = []
    for ep in data:
        pixels = ep["pixels"].permute(0, 2, 3, 1).numpy()
        future_pixels.append(pixels)
        if "state" in ep:
            state = ep["state"]
            if isinstance(state, torch.Tensor):
                state = state.numpy()
            future_states.append(state)
    future_pixels_bthwc = np.stack(future_pixels, axis=0)
    future_states_arr = np.stack(future_states, axis=0) if future_states else None
    future_latents = encode_pixels_sequence(ctx.model, future_pixels_bthwc, img_transform(ctx.cfg.img_size), ctx.device)
    start_latent = future_latents[:, 0, :]
    goal_latent = future_latents[:, -1, :]

    init_plus_goal = copy.deepcopy(init_step_np)
    init_plus_goal.update(copy.deepcopy(goal_step_np))
    shape_prefix = (int(ctx.world.num_envs), int(ctx.eval_cfg.world.history_size))
    init_step_broadcast = {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:]).copy()
        for k, v in init_plus_goal.items()
    }
    goal_step_broadcast = {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:]).copy()
        for k, v in goal_step_np.items()
    }

    batch = PreparedBatch(
        sampled_indices=sampled_indices,
        episodes_idx=eval_episodes,
        start_steps=eval_start_idx,
        data=data,
        init_step_np=init_step_np,
        goal_step_np=goal_step_np,
        init_step_broadcast=init_step_broadcast,
        goal_step_broadcast=goal_step_broadcast,
        future_pixels_bthwc=future_pixels_bthwc,
        future_states=future_states_arr,
        future_latents=future_latents,
        goal_latent=goal_latent,
        start_latent=start_latent,
    )
    reset_world_from_batch(ctx, batch)
    return batch


def reset_world_from_batch(ctx: ActingContext, batch: PreparedBatch) -> None:
    init_plus_goal = copy.deepcopy(batch.init_step_np)
    init_plus_goal.update(copy.deepcopy(batch.goal_step_np))
    seeds = batch.init_step_np.get("seed")
    variations_dict = {
        k.removeprefix("variation."): v
        for k, v in batch.init_step_np.items()
        if k.startswith("variation.")
    }
    options = [{} for _ in range(ctx.world.num_envs)]
    if len(variations_dict) > 0:
        for i in range(ctx.world.num_envs):
            options[i]["variation"] = list(variations_dict.keys())
            options[i]["variation_values"] = {k: v[i] for k, v in variations_dict.items()}

    ctx.world.reset(seed=seeds, options=options)

    callables = OmegaConf.to_container(ctx.eval_cfg.eval.get("callables"), resolve=True) or []
    for i, env in enumerate(ctx.world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        for spec in callables:
            method_name = spec["method"]
            if not hasattr(env_unwrapped, method_name):
                continue
            method = getattr(env_unwrapped, method_name)
            args = spec.get("args", spec)
            prepared_args = {}
            for arg_name, arg_data in args.items():
                value = arg_data.get("value", None)
                in_dataset = arg_data.get("in_dataset", True)
                if in_dataset:
                    if value not in init_plus_goal:
                        continue
                    prepared_args[arg_name] = copy.deepcopy(init_plus_goal[value][i])
                else:
                    prepared_args[arg_name] = arg_data.get("value")
            method(**prepared_args)

    ctx.world.infos.update(copy.deepcopy(batch.init_step_broadcast))
    ctx.world.infos.update(copy.deepcopy(batch.goal_step_broadcast))


def mse_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).mean(dim=-1)


def get_state_distance(info_dict: dict[str, Any]) -> float | None:
    if "state" not in info_dict or "goal_state" not in info_dict:
        return None
    state = np.asarray(info_dict["state"])
    goal = np.asarray(info_dict["goal_state"])
    if state.ndim >= 3:
        state = state[:, -1]
    if goal.ndim >= 3:
        goal = goal[:, -1]
    return float(np.linalg.norm(state - goal, axis=-1).mean())


def token_to_env_actions(
    tokens: torch.Tensor,
    *,
    process: dict[str, Any],
    raw_action_dim: int,
    group: int,
) -> np.ndarray:
    tokens_np = tokens.detach().cpu().numpy()
    if tokens_np.ndim != 3:
        raise ValueError(f"Expected token actions with ndim=3, got {tokens_np.shape}.")
    env_actions = tokens_np.reshape(tokens_np.shape[0], tokens_np.shape[1] * group, raw_action_dim)
    if "action" in process:
        flat = env_actions.reshape(-1, raw_action_dim)
        env_actions = process["action"].inverse_transform(flat).reshape(env_actions.shape)
    return env_actions


def action_token_summary(
    tokens: torch.Tensor,
    *,
    ref: ReferenceStats,
    process: dict[str, Any],
    raw_action_dim: int,
    group: int,
    env_action_space: Box,
) -> dict[str, Any]:
    x = tokens.reshape(-1, tokens.shape[-1])
    md2 = mahalanobis_sq(x, ref.mean, ref.inv_cov)
    zscores = ((x - ref.mean) / ref.std.clamp_min(1e-6)).abs()
    outside_q01_q99 = ((x < ref.q01) | (x > ref.q99)).float().mean()
    outside_q05_q95 = ((x < ref.q05) | (x > ref.q95)).float().mean()
    env_actions = token_to_env_actions(tokens, process=process, raw_action_dim=raw_action_dim, group=group)
    low = np.asarray(env_action_space.low)
    high = np.asarray(env_action_space.high)
    low_single = low[0] if low.ndim > 1 else low
    high_single = high[0] if high.ndim > 1 else high
    frac_out_bounds = float(((env_actions < low_single) | (env_actions > high_single)).mean())
    return {
        "mean_norm": float(x.norm(dim=-1).mean().item()),
        "std_norm": float(x.norm(dim=-1).std(unbiased=False).item()),
        "mahalanobis": stats_percentiles(md2),
        "mean_abs_zscore": float(zscores.mean().item()),
        "max_abs_zscore": float(zscores.max().item()),
        "fraction_outside_q01_q99": float(outside_q01_q99.item()),
        "fraction_outside_q05_q95": float(outside_q05_q95.item()),
        "fraction_outside_env_action_bounds": frac_out_bounds,
    }


def macro_action_summary(latents: torch.Tensor, ref: ReferenceStats) -> dict[str, Any]:
    x = latents.reshape(-1, latents.shape[-1])
    md2 = mahalanobis_sq(x, ref.mean, ref.inv_cov)
    return {
        "mean_norm": float(x.norm(dim=-1).mean().item()),
        "std_norm": float(x.norm(dim=-1).std(unbiased=False).item()),
        "mahalanobis": stats_percentiles(md2),
        "norm_ratio_vs_dataset": float(x.norm(dim=-1).mean().item() / ref.samples.norm(dim=-1).mean().item()),
    }


class InstrumentedStagePolicy(StagedHierarchicalWorldModelPolicy):
    def __init__(
        self,
        *,
        ctx: ActingContext,
        high_solver,
        low_solver,
        high_config,
        low_config,
        stage_duration_steps: int,
        clear_low_buffer_on_stage_change: bool,
        macro_replan_interval: int,
        high_latent_bounds: dict[str, np.ndarray] | None = None,
    ):
        super().__init__(
            model=ctx.model,
            high_solver=high_solver,
            low_solver=low_solver,
            high_config=high_config,
            low_config=low_config,
            stage_duration_steps=stage_duration_steps,
            clear_low_buffer_on_stage_change=clear_low_buffer_on_stage_change,
            macro_replan_interval=macro_replan_interval,
            process=ctx.process,
            transform=ctx.transform,
            high_latent_bounds=high_latent_bounds,
        )
        self.ctx = ctx
        self.high_plan_events: list[dict[str, Any]] = []
        self.low_block_events: list[dict[str, Any]] = []
        self.stage_end_events: list[dict[str, Any]] = []
        self.stage_end_latents: list[torch.Tensor] = []
        self.low_block_actual_latents: list[torch.Tensor] = []
        self.low_block_predicted_latents: list[torch.Tensor] = []
        self.low_block_target_latents: list[torch.Tensor] = []
        self.low_block_stage_indices: list[int] = []
        self.low_block_end_steps: list[int] = []
        self._pending_block: dict[str, Any] | None = None
        self._block_steps_remaining: int = 0
        self._stage_eval_done: set[int] = set()
        self._oracle_stage_targets: torch.Tensor | None = None
        self._stage_duration_schedule: list[int] | None = None
        self._generated_stage_targets: torch.Tensor | None = None

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        self.high_plan_events = []
        self.low_block_events = []
        self.stage_end_events = []
        self.stage_end_latents = []
        self.low_block_actual_latents = []
        self.low_block_predicted_latents = []
        self.low_block_target_latents = []
        self.low_block_stage_indices = []
        self.low_block_end_steps = []
        self._pending_block = None
        self._block_steps_remaining = 0
        self._stage_eval_done = set()
        self._generated_stage_targets = None

    def set_oracle_stage_targets(self, stage_targets: torch.Tensor, stage_durations_steps: Sequence[int]) -> None:
        self._oracle_stage_targets = stage_targets.detach().clone().to(next(self.model.parameters()).device)
        self._stage_duration_schedule = [int(x) for x in stage_durations_steps]

    def set_stage_duration_schedule(self, stage_durations_steps: Sequence[int]) -> None:
        self._stage_duration_schedule = [int(x) for x in stage_durations_steps]

    def _encode_current_latent(self, info_dict: dict[str, Any]) -> torch.Tensor:
        prepared = self._prepare_info({"pixels": np.array(info_dict["pixels"], copy=True)})
        pixels = prepared["pixels"].to(self._device())
        return self._encode_pixels_last(pixels)

    def _set_active_stage(self, stage_idx: int) -> None:
        prev = None if self._z_subgoal is None else self._z_subgoal.detach().clone()
        prev_stage_idx = None if self._z_subgoal is None else int(self._active_stage_idx)
        super()._set_active_stage(stage_idx)
        stage_changed = prev_stage_idx is None or int(self._active_stage_idx) != prev_stage_idx
        if stage_changed and self._z_subgoal is not None:
            churn = None
            if prev is not None:
                churn = float(mse_mean(prev, self._z_subgoal).mean().item())
            self.high_plan_events.append(
                {
                    "event": "stage_switch",
                    "stage_idx": int(self._active_stage_idx),
                    "step": int(self._steps_total),
                    "subgoal_mean_norm": float(self._z_subgoal.norm(dim=-1).mean().item()),
                    "subgoal_churn_mse": churn,
                }
            )

    def _sync_active_stage(self) -> None:
        if self._stage_targets is None:
            raise RuntimeError("Stage targets must be initialized before syncing stages.")
        if self._stage_duration_schedule:
            cumulative = np.cumsum(self._stage_duration_schedule)
            target_idx = int(np.searchsorted(cumulative, self._steps_total, side="right"))
            target_idx = min(target_idx, int(self._stage_targets.size(1)) - 1)
            self._set_active_stage(target_idx)
            return
        super()._sync_active_stage()

    @torch.inference_mode()
    def _plan_high(self, *, z_init: torch.Tensor, z_goal: torch.Tensor) -> None:
        if self._oracle_stage_targets is not None:
            self._stage_targets = self._oracle_stage_targets.to(z_init.device)
            self._steps_since_high = 0
            self._generated_stage_targets = None
            self._set_active_stage(0)
            return

        high_info = {
            "planner_level": "high",
            "z_init": z_init,
            "z_goal": z_goal,
        }
        outputs = self.high_solver(high_info, init_action=None)
        actions_solver = outputs["actions"]
        if not torch.is_tensor(actions_solver):
            actions_solver = torch.as_tensor(actions_solver)
        actions = actions_solver.to(z_init.device)

        full_h = int(self.high_cfg.horizon)
        high_plan = actions[:, :full_h]
        high_action_block = int(self.high_cfg.action_block)
        macro_seq = high_plan.reshape(
            z_init.size(0),
            full_h * high_action_block,
            self._latent_action_dim,
        )
        pred = self.model.rollout_high(z_init, macro_seq)
        if pred.ndim != 4 or pred.size(1) != 1:
            raise ValueError(
                "Expected rollout_high to return shape (B, 1, T, D) for staged planning, "
                f"got {tuple(pred.shape)}"
            )
        self._stage_targets = pred[:, 0]
        self._generated_stage_targets = self._stage_targets.detach().clone()
        self._steps_since_high = 0
        self.high_plan_events.append(
            {
                "event": "high_plan",
                "step": int(self._steps_total),
                "spans_tokens": [int(self.high_cfg.action_block)] * int(self.high_cfg.horizon),
                "macro_summary": macro_action_summary(
                    high_plan.reshape(-1, high_plan.shape[-1]).reshape(high_plan.shape[0], -1, high_plan.shape[-1]),
                    build_macro_reference(self.ctx, int(self.high_cfg.action_block)),
                ),
                "selected_high_plan_shape": list(high_plan.shape),
            }
        )
        self._set_active_stage(0)

    @torch.inference_mode()
    def _plan_low(self, *, z_init: torch.Tensor) -> None:
        if self._z_subgoal is None:
            raise RuntimeError("Low-level planning requested without a stage subgoal.")
        if self._low_grouped_action_dim is None:
            raise RuntimeError("Policy must be attached to env before planning.")

        n_envs = z_init.size(0)
        a_hist = torch.zeros(
            (n_envs, 1, self._low_grouped_action_dim),
            device=z_init.device,
            dtype=z_init.dtype,
        )
        low_info = {
            "planner_level": "low",
            "z_hist": z_init.unsqueeze(1),
            "a_hist": a_hist,
            "z_subgoal": self._z_subgoal,
        }

        low_init_action = self._next_low_init
        if torch.is_tensor(low_init_action):
            low_init_action = low_init_action.detach().cpu()

        outputs = self.low_solver(low_info, init_action=low_init_action)
        actions_solver = outputs["actions"]
        if not torch.is_tensor(actions_solver):
            actions_solver = torch.as_tensor(actions_solver)
        actions = actions_solver.to(z_init.device)

        keep_h = int(self.low_cfg.receding_horizon)
        plan = actions[:, :keep_h]
        self._next_low_init = (
            actions_solver[:, keep_h:].detach().cpu()
            if bool(getattr(self.low_cfg, "warm_start", True))
            else None
        )

        plan_env = plan.reshape(self.env.num_envs, self.flatten_receding_horizon_low, -1)
        self._action_buffer.extend(plan_env.transpose(0, 1))

        pred = self.model.rollout_low(
            z_init,
            torch.zeros((n_envs, 1, 1, self._low_grouped_action_dim), device=z_init.device, dtype=z_init.dtype),
            plan.unsqueeze(1),
        )[:, 0, -1, :]
        pred_err = mse_mean(pred, self._z_subgoal)
        self._pending_block = {
            "block_start_step": int(self._steps_total),
            "stage_idx": int(self._active_stage_idx),
            "predicted_terminal_latent": pred.detach().cpu(),
            "predicted_terminal_error_to_subgoal": pred_err.detach().cpu(),
            "selected_tokens": plan.detach().cpu(),
            "selected_action_summary": action_token_summary(
                plan,
                ref=self.ctx.action_ref,
                process=self.ctx.process,
                raw_action_dim=self.ctx.raw_action_dim,
                group=self.ctx.group,
                env_action_space=self.env.single_action_space,
            ),
        }
        self._block_steps_remaining = int(self.flatten_receding_horizon_low)

    def after_env_step(self, info_dict: dict[str, Any]) -> None:
        if self._pending_block is not None and self._block_steps_remaining == 0:
            z_actual = self._encode_current_latent(info_dict)
            pred = self._pending_block["predicted_terminal_latent"].to(z_actual.device)
            pred_err = self._pending_block["predicted_terminal_error_to_subgoal"].to(z_actual.device)
            actual_err = mse_mean(z_actual, self._z_subgoal)
            goal = self._prepare_info({"goal": np.array(info_dict["goal"], copy=True)})["goal"].to(self._device())
            z_goal = self._encode_pixels_last(goal)
            current_goal_err = mse_mean(z_actual, z_goal)
            self.low_block_events.append(
                {
                    "block_start_step": int(self._pending_block["block_start_step"]),
                    "block_end_step": int(self._steps_total),
                    "stage_idx": int(self._pending_block["stage_idx"]),
                    "model_error_mean": float(pred_err.mean().item()),
                    "actual_error_mean": float(actual_err.mean().item()),
                    "reality_gap_mean": float((actual_err - pred_err).mean().item()),
                    "goal_error_mean": float(current_goal_err.mean().item()),
                    "selected_action_summary": self._pending_block["selected_action_summary"],
                }
            )
            self.low_block_actual_latents.append(z_actual.detach().cpu())
            self.low_block_predicted_latents.append(self._pending_block["predicted_terminal_latent"].detach().cpu())
            self.low_block_target_latents.append(self._z_subgoal.detach().cpu())
            self.low_block_stage_indices.append(int(self._pending_block["stage_idx"]))
            self.low_block_end_steps.append(int(self._steps_total))
            self._pending_block = None

        if self._stage_duration_schedule is not None:
            cumulative = np.cumsum(self._stage_duration_schedule)
        else:
            num_stages = 0 if self._stage_targets is None else int(self._stage_targets.size(1))
            cumulative = np.cumsum([self.stage_duration_steps] * num_stages)

        for stage_idx, boundary in enumerate(cumulative):
            if int(self._steps_total) == int(boundary) and stage_idx not in self._stage_eval_done:
                z_actual = self._encode_current_latent(info_dict)
                if self._stage_targets is None:
                    break
                target = self._stage_targets[:, stage_idx, :]
                self.stage_end_events.append(
                    {
                        "stage_idx": int(stage_idx),
                        "stage_end_step": int(boundary),
                        "actual_error_mean": float(mse_mean(z_actual, target).mean().item()),
                    }
                )
                self.stage_end_latents.append(z_actual.detach().cpu())
                self._stage_eval_done.add(stage_idx)

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        action_np = super().get_action(info_dict, **kwargs)
        self._block_steps_remaining = max(0, self._block_steps_remaining - 1)
        return action_np


class InstrumentedHierarchicalPolicy(HierarchicalWorldModelPolicy):
    def __init__(
        self,
        *,
        ctx: ActingContext,
        high_solver,
        low_solver,
        high_config,
        low_config,
        macro_replan_interval: int,
        high_latent_bounds: dict[str, np.ndarray] | None = None,
    ):
        super().__init__(
            model=ctx.model,
            high_solver=high_solver,
            low_solver=low_solver,
            high_config=high_config,
            low_config=low_config,
            macro_replan_interval=macro_replan_interval,
            process=ctx.process,
            transform=ctx.transform,
            high_latent_bounds=high_latent_bounds,
        )
        self.ctx = ctx
        self.high_plan_events: list[dict[str, Any]] = []
        self.low_block_events: list[dict[str, Any]] = []
        self.step_events: list[dict[str, Any]] = []
        self.high_plan_current_latents: list[torch.Tensor] = []
        self.high_plan_goal_latents: list[torch.Tensor] = []
        self.high_plan_subgoal_latents: list[torch.Tensor] = []
        self.high_plan_steps: list[int] = []
        self.low_block_actual_latents: list[torch.Tensor] = []
        self.low_block_subgoal_latents: list[torch.Tensor] = []
        self.low_block_high_plan_ids: list[int] = []
        self.low_block_end_steps: list[int] = []
        self.step_current_latents: list[torch.Tensor] = []
        self.step_subgoal_latents: list[torch.Tensor] = []
        self.step_high_plan_ids: list[int] = []
        self._pending_block: dict[str, Any] | None = None
        self._block_steps_remaining: int = 0
        self._current_high_plan_id: int | None = None

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        self.high_plan_events = []
        self.low_block_events = []
        self.step_events = []
        self.high_plan_current_latents = []
        self.high_plan_goal_latents = []
        self.high_plan_subgoal_latents = []
        self.high_plan_steps = []
        self.low_block_actual_latents = []
        self.low_block_subgoal_latents = []
        self.low_block_high_plan_ids = []
        self.low_block_end_steps = []
        self.step_current_latents = []
        self.step_subgoal_latents = []
        self.step_high_plan_ids = []
        self._pending_block = None
        self._block_steps_remaining = 0
        self._current_high_plan_id = None

    def _encode_current_latent(self, info_dict: dict[str, Any]) -> torch.Tensor:
        prepared = self._prepare_info({"pixels": np.array(info_dict["pixels"], copy=True)})
        pixels = prepared["pixels"].to(self._device())
        return self._encode_pixels_last(pixels)

    @torch.inference_mode()
    def _plan_high(self, *, z_init: torch.Tensor, z_goal: torch.Tensor) -> None:
        prev_subgoal = None if self._z_subgoal is None else self._z_subgoal.detach().clone()
        high_info = {
            "planner_level": "high",
            "z_init": z_init,
            "z_goal": z_goal,
        }
        high_init_action = self._next_high_init
        if torch.is_tensor(high_init_action):
            high_init_action = high_init_action.detach().cpu()

        outputs = self.high_solver(high_info, init_action=high_init_action)
        actions_solver = outputs["actions"]
        if not torch.is_tensor(actions_solver):
            actions_solver = torch.as_tensor(actions_solver)
        actions = actions_solver.to(z_init.device)

        keep_h = int(self.high_cfg.receding_horizon)
        high_plan = actions[:, :keep_h]
        self._next_high_init = (
            actions_solver[:, keep_h:].detach().cpu()
            if bool(getattr(self.high_cfg, "warm_start", True))
            else None
        )
        high_action_block = int(self.high_cfg.action_block)
        macro_seq = high_plan.reshape(
            z_init.size(0),
            keep_h * high_action_block,
            self._latent_action_dim,
        )
        pred = self.model.rollout_high(z_init, macro_seq)
        self._z_subgoal = pred[:, 0, 0, :]
        self._steps_since_high = 0

        ref = build_macro_reference(self.ctx, keep_h * high_action_block)
        churn = None
        if prev_subgoal is not None:
            churn = float(mse_mean(prev_subgoal, self._z_subgoal).mean().item())
        plan_id = len(self.high_plan_events)
        self.high_plan_events.append(
            {
                "high_plan_id": plan_id,
                "step": int(self._steps_total if hasattr(self, "_steps_total") else 0),
                "subgoal_mean_norm": float(self._z_subgoal.norm(dim=-1).mean().item()),
                "subgoal_churn_mse": churn,
                "selected_macro_summary": macro_action_summary(macro_seq, ref),
            }
        )
        self.high_plan_current_latents.append(z_init.detach().cpu())
        self.high_plan_goal_latents.append(z_goal.detach().cpu())
        self.high_plan_subgoal_latents.append(self._z_subgoal.detach().cpu())
        self.high_plan_steps.append(int(self._steps_total if hasattr(self, "_steps_total") else 0))
        self._current_high_plan_id = plan_id

    @torch.inference_mode()
    def _plan_low(self, *, z_init: torch.Tensor) -> None:
        if self._z_subgoal is None:
            raise RuntimeError("Low-level planning requested without a high-level subgoal.")
        if self._low_grouped_action_dim is None:
            raise RuntimeError("Policy must be attached to env (set_env) before planning.")
        n_envs = z_init.size(0)
        a_hist = torch.zeros(
            (n_envs, 1, self._low_grouped_action_dim),
            device=z_init.device,
            dtype=z_init.dtype,
        )
        low_info = {
            "planner_level": "low",
            "z_hist": z_init.unsqueeze(1),
            "a_hist": a_hist,
            "z_subgoal": self._z_subgoal,
        }
        low_init_action = self._next_low_init
        if torch.is_tensor(low_init_action):
            low_init_action = low_init_action.detach().cpu()
        outputs = self.low_solver(low_info, init_action=low_init_action)
        actions_solver = outputs["actions"]
        if not torch.is_tensor(actions_solver):
            actions_solver = torch.as_tensor(actions_solver)
        actions = actions_solver.to(z_init.device)

        keep_h = int(self.low_cfg.receding_horizon)
        plan = actions[:, :keep_h]
        self._next_low_init = (
            actions_solver[:, keep_h:].detach().cpu()
            if bool(getattr(self.low_cfg, "warm_start", True))
            else None
        )
        plan_env = plan.reshape(self.env.num_envs, self.flatten_receding_horizon_low, -1)
        self._action_buffer.extend(plan_env.transpose(0, 1))

        pred = self.model.rollout_low(
            z_init,
            torch.zeros((n_envs, 1, 1, self._low_grouped_action_dim), device=z_init.device, dtype=z_init.dtype),
            plan.unsqueeze(1),
        )[:, 0, -1, :]
        pred_err = mse_mean(pred, self._z_subgoal)
        self._pending_block = {
            "block_start_step": int(self._steps_since_high),
            "high_plan_id": int(self._current_high_plan_id) if self._current_high_plan_id is not None else -1,
            "predicted_terminal_error_to_subgoal": pred_err.detach().cpu(),
            "selected_action_summary": action_token_summary(
                plan,
                ref=self.ctx.action_ref,
                process=self.ctx.process,
                raw_action_dim=self.ctx.raw_action_dim,
                group=self.ctx.group,
                env_action_space=self.env.single_action_space,
            ),
        }
        self._block_steps_remaining = int(self.flatten_receding_horizon_low)

    def after_env_step(self, info_dict: dict[str, Any]) -> None:
        prepared = self._prepare_info(
            {
                "pixels": np.array(info_dict["pixels"], copy=True),
                "goal": np.array(info_dict["goal"], copy=True),
            }
        )
        z_curr = self._encode_pixels_last(prepared["pixels"].to(self._device()))
        z_goal = self._encode_pixels_last(prepared["goal"].to(self._device()))
        if self._z_subgoal is None:
            z_subgoal_cpu = torch.full_like(z_curr.detach().cpu(), torch.nan)
        else:
            z_subgoal_cpu = self._z_subgoal.detach().cpu()
        self.step_current_latents.append(z_curr.detach().cpu())
        self.step_subgoal_latents.append(z_subgoal_cpu)
        self.step_high_plan_ids.append(int(self._current_high_plan_id) if self._current_high_plan_id is not None else -1)
        current_to_subgoal = None
        if self._z_subgoal is not None:
            current_to_subgoal = float(mse_mean(z_curr, self._z_subgoal).mean().item())
        self.step_events.append(
            {
                "env_step": len(self.step_events) + 1,
                "distance_current_to_subgoal": current_to_subgoal,
                "distance_current_to_final_goal": float(mse_mean(z_curr, z_goal).mean().item()),
            }
        )
        if self._pending_block is not None and self._block_steps_remaining == 0 and self._z_subgoal is not None:
            actual_err = mse_mean(z_curr, self._z_subgoal)
            pred_err = self._pending_block["predicted_terminal_error_to_subgoal"].to(z_curr.device)
            self.low_block_events.append(
                {
                    "block_end_step": len(self.step_events),
                    "model_error_mean": float(pred_err.mean().item()),
                    "actual_error_mean": float(actual_err.mean().item()),
                    "reality_gap_mean": float((actual_err - pred_err).mean().item()),
                    "selected_action_summary": self._pending_block["selected_action_summary"],
                }
            )
            self.low_block_actual_latents.append(z_curr.detach().cpu())
            self.low_block_subgoal_latents.append(self._z_subgoal.detach().cpu())
            self.low_block_high_plan_ids.append(int(self._pending_block["high_plan_id"]))
            self.low_block_end_steps.append(len(self.step_events))
            self._pending_block = None

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        action = super().get_action(info_dict, **kwargs)
        self._block_steps_remaining = max(0, self._block_steps_remaining - 1)
        return action


def build_solvers_and_configs(ctx: ActingContext):
    eval_cfg = copy.deepcopy(ctx.eval_cfg)
    eval_cfg.seed = int(ctx.cfg.seed)
    eval_cfg.planning.high.solver.device = str(ctx.device)
    eval_cfg.planning.low.solver.device = str(ctx.device)
    eval_cfg.planning.high.solver.num_samples = int(ctx.cfg.high_num_samples)
    eval_cfg.planning.high.solver.n_steps = int(ctx.cfg.high_iters)
    eval_cfg.planning.high.solver.topk = int(ctx.cfg.high_topk)
    eval_cfg.planning.low.solver.num_samples = int(ctx.cfg.low_num_samples)
    eval_cfg.planning.low.solver.n_steps = int(ctx.cfg.low_iters)
    eval_cfg.planning.low.solver.topk = int(ctx.cfg.low_topk)
    eval_cfg.planning.high.plan_config.horizon = int(ctx.cfg.high_horizon)
    eval_cfg.planning.high.plan_config.receding_horizon = 1
    eval_cfg.planning.high.plan_config.action_block = 1
    eval_cfg.planning.low.plan_config.horizon = int(ctx.cfg.low_horizon)
    eval_cfg.planning.low.plan_config.receding_horizon = int(ctx.cfg.low_receding_horizon)
    eval_cfg.planning.low.plan_config.action_block = int(ctx.cfg.frame_skip)
    high_cfg = swm.policy.PlanConfig(**eval_cfg.planning.high.plan_config)
    low_cfg = swm.policy.PlanConfig(**eval_cfg.planning.low.plan_config)
    high_solver_kwargs = OmegaConf.to_container(eval_cfg.planning.high.solver, resolve=True)
    low_solver_kwargs = OmegaConf.to_container(eval_cfg.planning.low.solver, resolve=True)
    high_solver_kwargs.pop("_target_", None)
    high_solver_kwargs.pop("model", None)
    low_solver_kwargs.pop("_target_", None)
    low_solver_kwargs.pop("model", None)
    high_solver = swm.solver.CEMSolver(model=ctx.model, **high_solver_kwargs)
    low_solver = swm.solver.CEMSolver(model=ctx.model, **low_solver_kwargs)
    return eval_cfg, high_cfg, low_cfg, high_solver, low_solver


def apply_goal_info(world: swm.World, batch: PreparedBatch) -> None:
    world.infos.update(copy.deepcopy(batch.goal_step_broadcast))


def run_world_loop(
    ctx: ActingContext,
    batch: PreparedBatch,
    policy,
    *,
    max_steps: int,
) -> dict[str, Any]:
    world = ctx.world
    world.set_policy(policy)
    episode_successes = np.zeros(world.num_envs, dtype=bool)
    for _ in range(int(max_steps)):
        apply_goal_info(world, batch)
        world.step()
        episode_successes = np.logical_or(episode_successes, world.terminateds)
        world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))
        if hasattr(policy, "after_env_step"):
            policy.after_env_step(world.infos)
    return {
        "episode_successes": episode_successes,
        "success_rate": float(episode_successes.mean() * 100.0),
    }


def nearest_future_offsets(
    generated_targets: torch.Tensor,
    future_latents: torch.Tensor,
    *,
    frame_skip: int,
) -> dict[str, Any]:
    b, num_stages, _ = generated_targets.shape
    future = future_latents[:, 1:, :]
    offsets_step = []
    offsets_token = []
    for step in range(num_stages):
        target = generated_targets[:, step : step + 1, :]
        d = torch.cdist(target, future).squeeze(1)
        nearest_idx = torch.argmin(d, dim=1) + 1
        offsets_step.append(nearest_idx.float())
        offsets_token.append(nearest_idx.float() / float(frame_skip))
    return {
        "nearest_future_offset_step_mean_per_stage": [float(x.mean().item()) for x in offsets_step],
        "nearest_future_offset_token_mean_per_stage": [float(x.mean().item()) for x in offsets_token],
        "nearest_future_offset_step_per_stage": [x.detach().cpu().tolist() for x in offsets_step],
        "nearest_future_offset_token_per_stage": [x.detach().cpu().tolist() for x in offsets_token],
    }


def future_index_for_offset(offset_steps: int, future_len: int) -> int:
    if future_len <= 0:
        raise ValueError("future_len must be positive.")
    return max(0, min(int(offset_steps) - 1, future_len - 1))


def stack_trace_latents(
    tensors: Sequence[torch.Tensor],
    *,
    num_envs: int,
    latent_dim: int,
) -> np.ndarray:
    if not tensors:
        return np.empty((0, num_envs, latent_dim), dtype=np.float32)
    return torch.stack([t.detach().cpu() for t in tensors], dim=0).numpy().astype(np.float32, copy=False)


def run_oracle_subgoal_acting(cfg: ActingDiagnosticConfig) -> dict[str, Any]:
    ctx = build_context(cfg)
    batch = prepare_eval_batch(ctx)
    eval_cfg, high_cfg, low_cfg, high_solver, low_solver = build_solvers_and_configs(ctx)
    stage_spans_tokens = partition_total(goal_tokens(cfg), int(cfg.high_horizon))
    stage_offsets_steps = np.cumsum([span * int(cfg.frame_skip) for span in stage_spans_tokens]).tolist()
    stage_durations_steps = np.diff([0] + stage_offsets_steps).tolist()
    future_len = int(batch.future_latents.shape[1])
    stage_targets = torch.stack(
        [batch.future_latents[:, future_index_for_offset(int(offset), future_len), :] for offset in stage_offsets_steps],
        dim=1,
    )
    policy = InstrumentedStagePolicy(
        ctx=ctx,
        high_solver=high_solver,
        low_solver=low_solver,
        high_config=high_cfg,
        low_config=low_cfg,
        stage_duration_steps=1,
        clear_low_buffer_on_stage_change=True,
        macro_replan_interval=int(eval_cfg.planning.high.replan_interval),
        high_latent_bounds=None,
    )
    policy.set_oracle_stage_targets(stage_targets, stage_durations_steps)
    reset_world_from_batch(ctx, batch)
    loop = run_world_loop(ctx, batch, policy, max_steps=int(cfg.goal_offset_steps))

    final_pixels = policy._encode_current_latent(ctx.world.infos)
    stage_end = {item["stage_idx"]: item for item in policy.stage_end_events}
    result = base_result(ctx)
    result.update(
        {
            "stage_offsets_steps": stage_offsets_steps,
            "stage_end_events": policy.stage_end_events,
            "low_block_events": policy.low_block_events,
            "success_rate": loop["success_rate"],
            "episode_successes": loop["episode_successes"].tolist(),
            "stage1_terminal_latent_error_mean": stage_end.get(0, {}).get("actual_error_mean"),
            "final_terminal_latent_error_mean": float(mse_mean(final_pixels, batch.goal_latent).mean().item()),
            "goal_progress_mean": float(
                (mse_mean(batch.start_latent, batch.goal_latent) - mse_mean(final_pixels, batch.goal_latent))
                .mean()
                .item()
            ),
            "low_level_block_model_error_mean": float(
                np.mean([item["model_error_mean"] for item in policy.low_block_events]) if policy.low_block_events else np.nan
            ),
            "low_level_block_actual_error_mean": float(
                np.mean([item["actual_error_mean"] for item in policy.low_block_events]) if policy.low_block_events else np.nan
            ),
            "reality_gap_mean": float(
                np.mean([item["reality_gap_mean"] for item in policy.low_block_events]) if policy.low_block_events else np.nan
            ),
        }
    )
    tsv_columns = [
        "experiment_kind",
        "policy",
        "goal_offset_steps",
        "high_horizon",
        "low_horizon",
        "low_receding_horizon",
        "stage1_terminal_latent_error_mean",
        "final_terminal_latent_error_mean",
        "goal_progress_mean",
        "reality_gap_mean",
        "success_rate",
        "result_status",
    ]
    tsv_row = {
        "experiment_kind": cfg.experiment_kind,
        "policy": cfg.policy,
        "goal_offset_steps": cfg.goal_offset_steps,
        "high_horizon": cfg.high_horizon,
        "low_horizon": cfg.low_horizon,
        "low_receding_horizon": cfg.low_receding_horizon,
        "stage1_terminal_latent_error_mean": result["stage1_terminal_latent_error_mean"],
        "final_terminal_latent_error_mean": result["final_terminal_latent_error_mean"],
        "goal_progress_mean": result["goal_progress_mean"],
        "reality_gap_mean": result["reality_gap_mean"],
        "success_rate": result["success_rate"],
        "result_status": "ok",
    }
    npz_arrays = {
        "sampled_indices": np.asarray(batch.sampled_indices),
        "episodes_idx": np.asarray(batch.episodes_idx),
        "start_steps": np.asarray(batch.start_steps),
        "episode_successes": np.asarray(loop["episode_successes"]),
        "future_latents": batch.future_latents.detach().cpu().numpy(),
        "start_latent": batch.start_latent.detach().cpu().numpy(),
        "goal_latent": batch.goal_latent.detach().cpu().numpy(),
        "oracle_stage_targets": stage_targets.detach().cpu().numpy(),
        "final_latent": final_pixels.detach().cpu().numpy(),
    }
    return finalize_result(ctx, result, tsv_columns=tsv_columns, tsv_row=tsv_row, npz_arrays=npz_arrays)


def run_low_level_reality_gap(cfg: ActingDiagnosticConfig) -> dict[str, Any]:
    ctx = build_context(cfg)
    batch = prepare_eval_batch(ctx)
    eval_cfg, high_cfg, low_cfg, high_solver, low_solver = build_solvers_and_configs(ctx)
    offsets = tuple(int(x) for x in cfg.subgoal_offsets)
    offset_results = []
    npz_arrays: dict[str, Any] = {
        "sampled_indices": np.asarray(batch.sampled_indices),
        "episodes_idx": np.asarray(batch.episodes_idx),
        "start_steps": np.asarray(batch.start_steps),
        "future_latents": batch.future_latents.detach().cpu().numpy(),
        "start_latent": batch.start_latent.detach().cpu().numpy(),
        "goal_latent": batch.goal_latent.detach().cpu().numpy(),
    }
    for offset in offsets:
        offset_steps = int(offset * cfg.frame_skip)
        future_idx = future_index_for_offset(offset_steps, int(batch.future_latents.shape[1]))
        stage_target = batch.future_latents[:, future_idx, :].unsqueeze(1)
        policy = InstrumentedStagePolicy(
            ctx=ctx,
            high_solver=high_solver,
            low_solver=low_solver,
            high_config=high_cfg,
            low_config=low_cfg,
            stage_duration_steps=1,
            clear_low_buffer_on_stage_change=True,
            macro_replan_interval=int(eval_cfg.planning.high.replan_interval),
            high_latent_bounds=None,
        )
        policy.set_oracle_stage_targets(stage_target, [offset_steps])
        reset_world_from_batch(ctx, batch)
        loop = run_world_loop(ctx, batch, policy, max_steps=offset_steps)
        z_final = policy._encode_current_latent(ctx.world.infos)
        actual_err = float(mse_mean(z_final, stage_target[:, 0, :]).mean().item())
        goal_progress = float(
            (mse_mean(batch.start_latent, batch.goal_latent) - mse_mean(z_final, batch.goal_latent)).mean().item()
        )
        result_item = {
            "offset_tokens": int(offset),
            "duration_steps": offset_steps,
            "model_error_mean": float(
                np.mean([item["model_error_mean"] for item in policy.low_block_events]) if policy.low_block_events else np.nan
            ),
            "actual_error_mean": actual_err,
            "reality_gap_mean": float(
                np.mean([item["reality_gap_mean"] for item in policy.low_block_events]) if policy.low_block_events else np.nan
            ),
            "goal_progress_mean": goal_progress,
            "success_rate": loop["success_rate"],
        }
        offset_results.append(result_item)
        npz_arrays[f"offset_{offset}_episode_successes"] = np.asarray(loop["episode_successes"])
    result = base_result(ctx)
    result.update(
        {
            "offset_results": offset_results,
            "overall_reality_gap_mean": float(np.mean([x["reality_gap_mean"] for x in offset_results])),
            "overall_actual_error_mean": float(np.mean([x["actual_error_mean"] for x in offset_results])),
        }
    )
    tsv_columns = [
        "experiment_kind",
        "policy",
        "goal_offset_steps",
        "low_horizon",
        "low_receding_horizon",
        "offset2_actual_error_mean",
        "offset3_actual_error_mean",
        "offset5_actual_error_mean",
        "overall_actual_error_mean",
        "overall_reality_gap_mean",
        "result_status",
    ]
    tsv_row = {
        "experiment_kind": cfg.experiment_kind,
        "policy": cfg.policy,
        "goal_offset_steps": cfg.goal_offset_steps,
        "low_horizon": cfg.low_horizon,
        "low_receding_horizon": cfg.low_receding_horizon,
        "overall_actual_error_mean": result["overall_actual_error_mean"],
        "overall_reality_gap_mean": result["overall_reality_gap_mean"],
        "result_status": "ok",
    }
    for item in offset_results:
        tsv_row[f"offset{item['offset_tokens']}_actual_error_mean"] = item["actual_error_mean"]
    return finalize_result(ctx, result, tsv_columns=tsv_columns, tsv_row=tsv_row, npz_arrays=npz_arrays)


def run_generated_subgoal_acting(cfg: ActingDiagnosticConfig) -> dict[str, Any]:
    ctx = build_context(cfg)
    batch = prepare_eval_batch(ctx)
    eval_cfg, high_cfg, low_cfg, high_solver, low_solver = build_solvers_and_configs(ctx)
    spans = partition_total(goal_tokens(cfg), int(cfg.high_horizon))
    stage_duration_schedule = [int(span * cfg.frame_skip) for span in spans]
    high_bounds = None
    if bool(eval_cfg.planning.high.latent_prior.get("enabled", True)):
        high_bounds = calibrate_latent_prior(
            model=ctx.model,
            dataset=ctx.dataset,
            cfg=eval_cfg.planning.high.latent_prior,
            process=ctx.process,
            seed=int(ctx.cfg.seed),
        )
    policy = InstrumentedStagePolicy(
        ctx=ctx,
        high_solver=high_solver,
        low_solver=low_solver,
        high_config=high_cfg,
        low_config=low_cfg,
        stage_duration_steps=1,
        clear_low_buffer_on_stage_change=True,
        macro_replan_interval=int(eval_cfg.planning.high.replan_interval),
        high_latent_bounds=high_bounds,
    )
    policy.set_stage_duration_schedule(stage_duration_schedule)
    reset_world_from_batch(ctx, batch)
    loop = run_world_loop(ctx, batch, policy, max_steps=int(cfg.goal_offset_steps))
    final_latent = policy._encode_current_latent(ctx.world.infos)
    generated_targets = policy._generated_stage_targets if policy._generated_stage_targets is not None else torch.empty(0)
    offset_info = nearest_future_offsets(generated_targets, batch.future_latents, frame_skip=int(cfg.frame_skip))
    expected_tokens = np.cumsum(spans).tolist()
    stage_results = {item["stage_idx"]: item for item in policy.stage_end_events}
    step_results = []
    for idx, expected in enumerate(expected_tokens):
        step_results.append(
            {
                "step": int(idx + 1),
                "expected_offset_token": int(expected),
                "nearest_future_offset_token_mean": offset_info["nearest_future_offset_token_mean_per_stage"][idx],
                "offset_error_token_mean": float(
                    offset_info["nearest_future_offset_token_mean_per_stage"][idx] - expected
                ),
                "stage_end_actual_error_mean": stage_results.get(idx, {}).get("actual_error_mean"),
            }
        )
    result = base_result(ctx)
    result.update(
        {
            "spans_tokens": spans,
            "step_results": step_results,
            "stage_end_events": policy.stage_end_events,
            "low_block_events": policy.low_block_events,
            "high_plan_events": policy.high_plan_events,
            "success_rate": loop["success_rate"],
            "episode_successes": loop["episode_successes"].tolist(),
            "final_terminal_latent_error_mean": float(mse_mean(final_latent, batch.goal_latent).mean().item()),
            "goal_progress_mean": float(
                (mse_mean(batch.start_latent, batch.goal_latent) - mse_mean(final_latent, batch.goal_latent))
                .mean()
                .item()
            ),
            "offset_info": offset_info,
        }
    )
    tsv_columns = [
        "experiment_kind",
        "policy",
        "goal_offset_steps",
        "high_horizon",
        "low_horizon",
        "low_receding_horizon",
        "step1_stage_end_actual_error_mean",
        "step1_offset_error_token_mean",
        "step2_stage_end_actual_error_mean",
        "step2_offset_error_token_mean",
        "final_terminal_latent_error_mean",
        "success_rate",
        "result_status",
    ]
    tsv_row = {
        "experiment_kind": cfg.experiment_kind,
        "policy": cfg.policy,
        "goal_offset_steps": cfg.goal_offset_steps,
        "high_horizon": cfg.high_horizon,
        "low_horizon": cfg.low_horizon,
        "low_receding_horizon": cfg.low_receding_horizon,
        "final_terminal_latent_error_mean": result["final_terminal_latent_error_mean"],
        "success_rate": result["success_rate"],
        "result_status": "ok",
    }
    for item in step_results[:2]:
        tsv_row[f"step{item['step']}_stage_end_actual_error_mean"] = item["stage_end_actual_error_mean"]
        tsv_row[f"step{item['step']}_offset_error_token_mean"] = item["offset_error_token_mean"]
    npz_arrays = {
        "sampled_indices": np.asarray(batch.sampled_indices),
        "episodes_idx": np.asarray(batch.episodes_idx),
        "start_steps": np.asarray(batch.start_steps),
        "generated_stage_targets": generated_targets.detach().cpu().numpy(),
        "future_latents": batch.future_latents.detach().cpu().numpy(),
        "start_latent": batch.start_latent.detach().cpu().numpy(),
        "goal_latent": batch.goal_latent.detach().cpu().numpy(),
        "final_latent": final_latent.detach().cpu().numpy(),
        "episode_successes": np.asarray(loop["episode_successes"]),
        "stage_end_actual_latents": stack_trace_latents(
            policy.stage_end_latents,
            num_envs=int(batch.start_latent.shape[0]),
            latent_dim=int(batch.start_latent.shape[1]),
        ),
        "low_block_actual_latents": stack_trace_latents(
            policy.low_block_actual_latents,
            num_envs=int(batch.start_latent.shape[0]),
            latent_dim=int(batch.start_latent.shape[1]),
        ),
        "low_block_predicted_latents": stack_trace_latents(
            policy.low_block_predicted_latents,
            num_envs=int(batch.start_latent.shape[0]),
            latent_dim=int(batch.start_latent.shape[1]),
        ),
        "low_block_target_latents": stack_trace_latents(
            policy.low_block_target_latents,
            num_envs=int(batch.start_latent.shape[0]),
            latent_dim=int(batch.start_latent.shape[1]),
        ),
        "low_block_stage_indices": np.asarray(policy.low_block_stage_indices, dtype=np.int64),
        "low_block_end_steps": np.asarray(policy.low_block_end_steps, dtype=np.int64),
    }
    return finalize_result(ctx, result, tsv_columns=tsv_columns, tsv_row=tsv_row, npz_arrays=npz_arrays)


def run_online_hierarchical_logging(cfg: ActingDiagnosticConfig) -> dict[str, Any]:
    ctx = build_context(cfg)
    batch = prepare_eval_batch(ctx)
    eval_cfg, high_cfg, low_cfg, high_solver, low_solver = build_solvers_and_configs(ctx)
    high_bounds = None
    if bool(eval_cfg.planning.high.latent_prior.get("enabled", True)):
        high_bounds = calibrate_latent_prior(
            model=ctx.model,
            dataset=ctx.dataset,
            cfg=eval_cfg.planning.high.latent_prior,
            process=ctx.process,
            seed=int(ctx.cfg.seed),
        )
    policy = InstrumentedHierarchicalPolicy(
        ctx=ctx,
        high_solver=high_solver,
        low_solver=low_solver,
        high_config=high_cfg,
        low_config=low_cfg,
        macro_replan_interval=int(eval_cfg.planning.high.replan_interval),
        high_latent_bounds=high_bounds,
    )
    reset_world_from_batch(ctx, batch)
    loop = run_world_loop(ctx, batch, policy, max_steps=int(cfg.eval_budget))
    final_latent = policy._encode_current_latent(ctx.world.infos)
    subgoal_distances = [item["distance_current_to_subgoal"] for item in policy.step_events if item["distance_current_to_subgoal"] is not None]
    goal_distances = [item["distance_current_to_final_goal"] for item in policy.step_events]
    churn_values = [item["subgoal_churn_mse"] for item in policy.high_plan_events if item.get("subgoal_churn_mse") is not None]
    result = base_result(ctx)
    result.update(
        {
            "success_rate": loop["success_rate"],
            "episode_successes": loop["episode_successes"].tolist(),
            "high_plan_events": policy.high_plan_events,
            "low_block_events": policy.low_block_events,
            "step_events": policy.step_events,
            "final_terminal_latent_error_mean": float(mse_mean(final_latent, batch.goal_latent).mean().item()),
            "mean_distance_current_to_subgoal": float(np.mean(subgoal_distances)) if subgoal_distances else None,
            "mean_distance_current_to_final_goal": float(np.mean(goal_distances)) if goal_distances else None,
            "mean_subgoal_churn_mse": float(np.mean(churn_values)) if churn_values else None,
            "mean_reality_gap": float(
                np.mean([item["reality_gap_mean"] for item in policy.low_block_events]) if policy.low_block_events else np.nan
            ),
        }
    )
    tsv_columns = [
        "experiment_kind",
        "policy",
        "goal_offset_steps",
        "high_horizon",
        "low_horizon",
        "low_receding_horizon",
        "success_rate",
        "final_terminal_latent_error_mean",
        "mean_distance_current_to_subgoal",
        "mean_distance_current_to_final_goal",
        "mean_subgoal_churn_mse",
        "mean_reality_gap",
        "result_status",
    ]
    tsv_row = {
        "experiment_kind": cfg.experiment_kind,
        "policy": cfg.policy,
        "goal_offset_steps": cfg.goal_offset_steps,
        "high_horizon": cfg.high_horizon,
        "low_horizon": cfg.low_horizon,
        "low_receding_horizon": cfg.low_receding_horizon,
        "success_rate": result["success_rate"],
        "final_terminal_latent_error_mean": result["final_terminal_latent_error_mean"],
        "mean_distance_current_to_subgoal": result["mean_distance_current_to_subgoal"],
        "mean_distance_current_to_final_goal": result["mean_distance_current_to_final_goal"],
        "mean_subgoal_churn_mse": result["mean_subgoal_churn_mse"],
        "mean_reality_gap": result["mean_reality_gap"],
        "result_status": "ok",
    }
    npz_arrays = {
        "sampled_indices": np.asarray(batch.sampled_indices),
        "episodes_idx": np.asarray(batch.episodes_idx),
        "start_steps": np.asarray(batch.start_steps),
        "episode_successes": np.asarray(loop["episode_successes"]),
        "future_latents": batch.future_latents.detach().cpu().numpy(),
        "start_latent": batch.start_latent.detach().cpu().numpy(),
        "goal_latent": batch.goal_latent.detach().cpu().numpy(),
        "final_latent": final_latent.detach().cpu().numpy(),
        "high_plan_steps": np.asarray(policy.high_plan_steps, dtype=np.int64),
        "high_plan_current_latents": stack_trace_latents(
            policy.high_plan_current_latents,
            num_envs=int(batch.start_latent.shape[0]),
            latent_dim=int(batch.start_latent.shape[1]),
        ),
        "high_plan_goal_latents": stack_trace_latents(
            policy.high_plan_goal_latents,
            num_envs=int(batch.start_latent.shape[0]),
            latent_dim=int(batch.start_latent.shape[1]),
        ),
        "high_plan_subgoal_latents": stack_trace_latents(
            policy.high_plan_subgoal_latents,
            num_envs=int(batch.start_latent.shape[0]),
            latent_dim=int(batch.start_latent.shape[1]),
        ),
        "low_block_end_steps": np.asarray(policy.low_block_end_steps, dtype=np.int64),
        "low_block_high_plan_ids": np.asarray(policy.low_block_high_plan_ids, dtype=np.int64),
        "low_block_actual_latents": stack_trace_latents(
            policy.low_block_actual_latents,
            num_envs=int(batch.start_latent.shape[0]),
            latent_dim=int(batch.start_latent.shape[1]),
        ),
        "low_block_subgoal_latents": stack_trace_latents(
            policy.low_block_subgoal_latents,
            num_envs=int(batch.start_latent.shape[0]),
            latent_dim=int(batch.start_latent.shape[1]),
        ),
        "step_current_latents": stack_trace_latents(
            policy.step_current_latents,
            num_envs=int(batch.start_latent.shape[0]),
            latent_dim=int(batch.start_latent.shape[1]),
        ),
        "step_subgoal_latents": stack_trace_latents(
            policy.step_subgoal_latents,
            num_envs=int(batch.start_latent.shape[0]),
            latent_dim=int(batch.start_latent.shape[1]),
        ),
        "step_high_plan_ids": np.asarray(policy.step_high_plan_ids, dtype=np.int64),
    }
    return finalize_result(ctx, result, tsv_columns=tsv_columns, tsv_row=tsv_row, npz_arrays=npz_arrays)


def run_acting_diagnostic(cfg: ActingDiagnosticConfig) -> dict[str, Any]:
    if cfg.experiment_kind == "oracle_subgoal_acting":
        return run_oracle_subgoal_acting(cfg)
    if cfg.experiment_kind == "low_level_reality_gap":
        return run_low_level_reality_gap(cfg)
    if cfg.experiment_kind == "generated_subgoal_acting":
        return run_generated_subgoal_acting(cfg)
    if cfg.experiment_kind == "online_hierarchical_logging":
        return run_online_hierarchical_logging(cfg)
    raise ValueError(f"Unknown experiment kind: {cfg.experiment_kind}")
