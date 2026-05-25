from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np
import torch
import warnings
from torchvision import tv_tensors

try:
    from gymnasium.spaces import Box
except Exception:  # pragma: no cover - fallback for environments without gymnasium
    class Box:  # type: ignore[override]
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.shape = tuple(shape or self.low.shape)
            self.dtype = dtype


def _to_torch(x: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    raise ValueError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")


def _infer_latent_dim_from_model(model: torch.nn.Module) -> int:
    if hasattr(model, "_infer_latent_action_dim"):
        return int(model._infer_latent_action_dim())  # type: ignore[attr-defined]
    if hasattr(model, "latent_action_encoder"):
        latent_encoder = model.latent_action_encoder
        if hasattr(latent_encoder, "output_proj"):
            return int(latent_encoder.output_proj.out_features)
        if hasattr(latent_encoder, "latent_dim"):
            return int(latent_encoder.latent_dim)
    raise ValueError("Unable to infer latent action dimension from model.")


def _infer_macro_input_dim_from_model(model: torch.nn.Module) -> int | None:
    """Infer latent-action encoder token input dimension if available."""
    if not hasattr(model, "latent_action_encoder"):
        return None
    latent_encoder = model.latent_action_encoder
    if hasattr(latent_encoder, "input_proj") and hasattr(latent_encoder.input_proj, "in_features"):
        return int(latent_encoder.input_proj.in_features)
    return None


def _try_get_episode_metadata(dataset) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Best-effort fetch of per-row episode and step metadata.

    Returns:
        Tuple (episode_ids, step_idx) where each entry is a 1D numpy array
        aligned with dataset rows, or None when unavailable.
    """
    if not hasattr(dataset, "get_col_data"):
        return None, None

    episode_ids = None
    column_names = set(getattr(dataset, "column_names", []) or [])

    preferred_episode_cols: list[str] = []
    if "episode_idx" in column_names:
        preferred_episode_cols.append("episode_idx")
    if "ep_idx" in column_names:
        preferred_episode_cols.append("ep_idx")
    if not preferred_episode_cols:
        preferred_episode_cols = ["episode_idx", "ep_idx"]

    for col in preferred_episode_cols:
        try:
            col_data = dataset.get_col_data(col)
        except Exception:
            continue
        if col_data is None:
            continue
        episode_ids = np.asarray(col_data)
        break

    if episode_ids is None:
        return None, None

    step_idx = None
    try:
        step_data = dataset.get_col_data("step_idx")
        if step_data is not None:
            step_idx = np.asarray(step_data)
    except Exception:
        step_idx = None

    return episode_ids, step_idx


def _sample_valid_chunk_starts(
    *,
    episode_ids: np.ndarray | None,
    step_idx: np.ndarray | None,
    seq_len: int,
    chunk_len: int,
    num_chunks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample chunk starts that do not cross episode boundaries.

    If episode metadata is unavailable, falls back to uniform starts over the
    flattened sequence, preserving previous behavior.
    """
    if chunk_len <= 0 or seq_len < chunk_len:
        return np.empty((0,), dtype=np.int64)

    max_start = seq_len - chunk_len
    if episode_ids is None:
        return rng.integers(0, max_start + 1, size=num_chunks)

    # Mark boundary violations between consecutive rows.
    ep_change = episode_ids[1:] != episode_ids[:-1]
    bad_transition = ep_change.copy()
    if step_idx is not None:
        bad_transition |= (step_idx[1:] - step_idx[:-1]) != 1

    # A start is valid when all transitions inside the chunk are valid.
    transitions_per_chunk = chunk_len - 1
    if transitions_per_chunk <= 0:
        valid_starts = np.arange(max_start + 1, dtype=np.int64)
    else:
        bad_i64 = bad_transition.astype(np.int64)
        csum = np.cumsum(np.concatenate(([0], bad_i64)))
        window_bad = csum[transitions_per_chunk:] - csum[:-transitions_per_chunk]
        valid_starts = np.nonzero(window_bad == 0)[0].astype(np.int64)

    if valid_starts.size == 0:
        return np.empty((0,), dtype=np.int64)

    replace = valid_starts.size < num_chunks
    idx = rng.choice(valid_starts.size, size=num_chunks, replace=replace)
    return valid_starts[idx]


@torch.inference_mode()
def build_empirical_macro_action_bank(
    *,
    model: torch.nn.Module,
    dataset,
    cfg,
    high_horizon: int,
    high_action_block: int,
    process: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Encode train-set macro-action sequences for empirical high-level search."""
    latent_dim = _infer_latent_dim_from_model(model)
    if not bool(cfg.get("enabled", False)):
        return {
            "actions": np.empty((0, int(high_horizon), int(high_action_block) * latent_dim), dtype=np.float32),
            "num_sequences": np.array(0, dtype=np.int64),
        }
    if not hasattr(dataset, "get_col_data"):
        raise ValueError("Empirical macro-action search requires a dataset with get_col_data.")
    if not hasattr(model, "encode_macro_actions"):
        raise ValueError("Empirical macro-action search requires model.encode_macro_actions.")

    action_data = dataset.get_col_data("action")
    if action_data is None:
        raise ValueError("Empirical macro-action search requires dataset action data.")
    action_data = np.asarray(action_data)
    if action_data.ndim != 2 or action_data.shape[1] <= 0:
        raise ValueError(f"Expected action data with shape (T, A), got {action_data.shape}.")

    chunk_len = int(cfg.get("chunk_len", 5))
    num_sequences = int(cfg.get("num_sequences", 4096))
    high_horizon = int(high_horizon)
    high_action_block = int(high_action_block)
    total_macro_tokens = high_horizon * high_action_block
    if chunk_len <= 0 or num_sequences <= 0 or total_macro_tokens <= 0:
        raise ValueError("Empirical macro-action chunk_len, num_sequences, and horizon must be positive.")

    valid_mask = ~np.isnan(action_data).any(axis=1)
    valid_row_idx = np.nonzero(valid_mask)[0]
    action_data = action_data[valid_mask]
    episode_ids, step_idx = _try_get_episode_metadata(dataset)
    if episode_ids is not None:
        episode_ids = episode_ids[valid_row_idx]
    if step_idx is not None:
        step_idx = step_idx[valid_row_idx]

    if process and "action" in process:
        action_data = process["action"].transform(action_data)

    raw_action_dim = int(action_data.shape[1])
    expected_action_dim = _infer_macro_input_dim_from_model(model)
    if expected_action_dim is None:
        expected_action_dim = raw_action_dim
    if expected_action_dim < raw_action_dim or expected_action_dim % raw_action_dim != 0:
        raise ValueError(
            "Unable to match action dimension for empirical macro-action bank: "
            f"dataset action dim={raw_action_dim}, model latent-action input dim={expected_action_dim}."
        )

    raw_group = expected_action_dim // raw_action_dim
    raw_macro_len = chunk_len * raw_group
    total_raw_len = total_macro_tokens * raw_macro_len
    starts = _sample_valid_chunk_starts(
        episode_ids=episode_ids,
        step_idx=step_idx,
        seq_len=action_data.shape[0],
        chunk_len=total_raw_len,
        num_chunks=num_sequences,
        rng=np.random.default_rng(seed),
    )
    if starts.size == 0:
        raise ValueError(
            "No valid contiguous action spans found for empirical macro-action bank "
            f"(total_raw_len={total_raw_len})."
        )

    raw_sequences = np.stack([action_data[s : s + total_raw_len] for s in starts], axis=0)
    grouped = raw_sequences.reshape(num_sequences, total_macro_tokens, chunk_len, expected_action_dim)
    flat_chunks = grouped.reshape(num_sequences * total_macro_tokens, chunk_len, expected_action_dim)

    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    encode_batch_size = max(1, int(cfg.get("encode_batch_size", 4096)))
    encoded_batches: list[np.ndarray] = []
    flat_chunks = flat_chunks.astype(np.float32, copy=False)
    for start in range(0, flat_chunks.shape[0], encode_batch_size):
        chunks_t = torch.from_numpy(flat_chunks[start : start + encode_batch_size]).to(device)
        mask_t = torch.ones((chunks_t.size(0), chunks_t.size(1)), dtype=torch.bool, device=device)
        encoded_batches.append(model.encode_macro_actions(chunks_t, mask_t).detach().cpu().numpy())
    latents = np.concatenate(encoded_batches, axis=0)
    if latents.ndim != 2 or latents.shape[1] != latent_dim:
        raise ValueError(f"Unexpected encoded macro-action shape: {latents.shape}.")

    actions = latents.reshape(num_sequences, high_horizon, high_action_block * latent_dim)
    row_indices = valid_row_idx[starts]
    goal_row_indices = valid_row_idx[starts + total_raw_len - 1]
    return {
        "actions": actions.astype(np.float32),
        "row_indices": row_indices.astype(np.int64),
        "goal_row_indices": goal_row_indices.astype(np.int64),
        "num_sequences": np.array(num_sequences, dtype=np.int64),
        "chunk_len": np.array(chunk_len, dtype=np.int64),
        "raw_macro_len": np.array(raw_macro_len, dtype=np.int64),
        "encode_batch_size": np.array(encode_batch_size, dtype=np.int64),
    }


class EmpiricalMacroActionSolver:
    """High-level solver constrained to train-set macro-action sequences.

    Candidates are sampled as empirical_macro_sequence + residual. The returned
    action is always the best actual candidate observed, not the residual mean.
    """

    def __init__(
        self,
        *,
        model,
        macro_bank: np.ndarray | torch.Tensor,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1.0,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        residual_scale: float = 0.1,
        min_residual_std: float = 1.0e-3,
        return_top_candidates: int = 8,
        stage_sampling: str = "sequence",
    ) -> None:
        self.model = model
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.var_scale = float(var_scale)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.device = torch.device(device)
        self.residual_scale = float(residual_scale)
        self.min_residual_std = float(min_residual_std)
        self.return_top_candidates = int(return_top_candidates)
        self.stage_sampling = str(stage_sampling)
        if self.stage_sampling not in {"sequence", "independent"}:
            raise ValueError("stage_sampling must be one of: sequence, independent.")
        self.torch_gen = torch.Generator(device=self.device).manual_seed(int(seed))
        self.macro_bank = torch.as_tensor(macro_bank, dtype=torch.float32)

    def configure(self, *, action_space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True
        expected_shape = (int(config.horizon), self.action_dim)
        if self.macro_bank.ndim != 3 or tuple(self.macro_bank.shape[1:]) != expected_shape:
            raise ValueError(
                "Empirical macro bank must have shape (N, horizon, action_dim), "
                f"expected (*, {expected_shape[0]}, {expected_shape[1]}), "
                f"got {tuple(self.macro_bank.shape)}."
            )

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * int(self._config.action_block)

    @property
    def horizon(self) -> int:
        return int(self._config.horizon)

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def _sample_base_indices(
        self,
        *,
        current_bs: int,
        bank_size: int,
    ) -> torch.Tensor:
        index_shape = (
            (current_bs, self.num_samples, self.horizon)
            if self.stage_sampling == "independent"
            else (current_bs, self.num_samples)
        )
        return torch.randint(
            bank_size,
            index_shape,
            generator=self.torch_gen,
            device=self.device,
        )

    def _gather_base_actions(self, bank: torch.Tensor, base_idx: torch.Tensor) -> torch.Tensor:
        if self.stage_sampling == "sequence":
            return bank[base_idx]
        stage_idx = torch.arange(self.horizon, device=self.device).view(1, 1, self.horizon)
        stage_idx = stage_idx.expand(base_idx.shape)
        return bank[base_idx, stage_idx]

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        import time

        del init_action
        start_time = time.time()
        bank = self.macro_bank.to(self.device)
        if bank.size(0) <= 0:
            raise ValueError("Empirical macro bank is empty.")

        outputs = {"costs": []}
        actions_out = torch.zeros(self.n_envs, self.horizon, self.action_dim, device=self.device)
        top_n = max(1, min(self.return_top_candidates, self.num_samples, self.topk))
        top_actions_out = torch.zeros(
            self.n_envs,
            top_n,
            self.horizon,
            self.action_dim,
            device=self.device,
        )
        top_costs_out = torch.full((self.n_envs, top_n), float("inf"), device=self.device)

        for start_idx in range(0, self.n_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_envs)
            current_bs = end_idx - start_idx
            residual_mean = torch.zeros(
                current_bs,
                self.horizon,
                self.action_dim,
                device=self.device,
            )
            residual_std = torch.full_like(residual_mean, self.residual_scale * self.var_scale)

            expanded_infos = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    v_batch = v[start_idx:end_idx].to(self.device)
                    v_batch = v_batch.unsqueeze(1).expand(current_bs, self.num_samples, *v_batch.shape[1:])
                elif isinstance(v, np.ndarray):
                    v_batch = v[start_idx:end_idx]
                    v_batch = np.repeat(v_batch[:, None, ...], self.num_samples, axis=1)
                else:
                    v_batch = v
                expanded_infos[k] = v_batch

            best_costs = torch.full((current_bs, top_n), float("inf"), device=self.device)
            best_actions = torch.zeros(
                current_bs,
                top_n,
                self.horizon,
                self.action_dim,
                device=self.device,
            )

            for _ in range(self.n_steps):
                base_idx = self._sample_base_indices(
                    current_bs=current_bs,
                    bank_size=bank.size(0),
                )
                base = self._gather_base_actions(bank, base_idx)
                residual = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                )
                residual = residual * residual_std.unsqueeze(1) + residual_mean.unsqueeze(1)
                residual[:, 0] = 0.0
                candidates = base + residual
                costs = self.model.get_cost(expanded_infos.copy(), candidates)

                _, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                elite_actions = candidates[batch_indices, topk_inds]
                elite_base = base[batch_indices, topk_inds]
                elite_residual = elite_actions - elite_base
                residual_mean = elite_residual.mean(dim=1)
                residual_std = elite_residual.std(dim=1, unbiased=False).clamp_min(self.min_residual_std)

                combined_costs = torch.cat([best_costs, costs], dim=1)
                combined_actions = torch.cat([best_actions, candidates], dim=1)
                best_costs, best_inds = torch.topk(combined_costs, k=top_n, dim=1, largest=False)
                gather_idx = best_inds[:, :, None, None].expand(-1, -1, self.horizon, self.action_dim)
                best_actions = torch.gather(combined_actions, dim=1, index=gather_idx)

            actions_out[start_idx:end_idx] = best_actions[:, 0]
            top_actions_out[start_idx:end_idx] = best_actions
            top_costs_out[start_idx:end_idx] = best_costs
            outputs["costs"].extend(best_costs[:, 0].detach().cpu().tolist())

        outputs["actions"] = actions_out.detach().cpu()
        outputs["candidate_actions"] = top_actions_out.detach().cpu()
        outputs["candidate_costs"] = top_costs_out.detach().cpu()
        print(f"Empirical macro solve time: {time.time() - start_time:.4f} seconds")
        return outputs


@torch.inference_mode()
def calibrate_latent_prior(
    *,
    model: torch.nn.Module,
    dataset,
    cfg,
    process: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Estimate latent macro-action bounds from dataset action chunks.

    Returns:
        dict with:
          - low: (D_l,) float32
          - high: (D_l,) float32
          - num_chunks: scalar int64 array
          - chunk_len: scalar int64 array
    """
    latent_dim = _infer_latent_dim_from_model(model)
    fallback_abs = float(cfg.get("fallback_abs", 1.0))
    fallback_low = np.full((latent_dim,), -fallback_abs, dtype=np.float32)
    fallback_high = np.full((latent_dim,), fallback_abs, dtype=np.float32)

    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        return {
            "low": fallback_low,
            "high": fallback_high,
            "num_chunks": np.array(0, dtype=np.int64),
            "chunk_len": np.array(0, dtype=np.int64),
        }

    if not hasattr(dataset, "get_col_data"):
        return {
            "low": fallback_low,
            "high": fallback_high,
            "num_chunks": np.array(0, dtype=np.int64),
            "chunk_len": np.array(0, dtype=np.int64),
        }

    action_data = dataset.get_col_data("action")
    if action_data is None:
        return {
            "low": fallback_low,
            "high": fallback_high,
            "num_chunks": np.array(0, dtype=np.int64),
            "chunk_len": np.array(0, dtype=np.int64),
        }

    action_data = np.asarray(action_data)
    if action_data.ndim != 2 or action_data.shape[1] <= 0:
        return {
            "low": fallback_low,
            "high": fallback_high,
            "num_chunks": np.array(0, dtype=np.int64),
            "chunk_len": np.array(0, dtype=np.int64),
        }

    chunk_len = int(cfg.get("chunk_len", 5))
    num_chunks = int(cfg.get("num_chunks", 2048))
    min_chunks = int(cfg.get("min_chunks_for_stats", 64))
    q_low = float(cfg.get("lower_q", 5.0))
    q_high = float(cfg.get("upper_q", 95.0))
    margin_ratio = float(cfg.get("margin_ratio", 0.05))

    if chunk_len <= 0 or num_chunks <= 0 or action_data.shape[0] < chunk_len:
        return {
            "low": fallback_low,
            "high": fallback_high,
            "num_chunks": np.array(0, dtype=np.int64),
            "chunk_len": np.array(chunk_len, dtype=np.int64),
        }

    # Drop rows with NaNs and keep metadata aligned with remaining rows.
    valid_mask = ~np.isnan(action_data).any(axis=1)
    valid_row_idx = np.nonzero(valid_mask)[0]
    action_data = action_data[valid_mask]
    if action_data.shape[0] < chunk_len:
        return {
            "low": fallback_low,
            "high": fallback_high,
            "num_chunks": np.array(0, dtype=np.int64),
            "chunk_len": np.array(chunk_len, dtype=np.int64),
        }

    episode_ids, step_idx = _try_get_episode_metadata(dataset)
    if episode_ids is not None:
        episode_ids = episode_ids[valid_row_idx]
    if step_idx is not None:
        step_idx = step_idx[valid_row_idx]

    # Apply action normalization in raw action space first.
    if process and "action" in process:
        proc = process["action"]
        action_data = proc.transform(action_data)

    raw_action_dim = int(action_data.shape[1])
    expected_action_dim = _infer_macro_input_dim_from_model(model)
    if expected_action_dim is None:
        expected_action_dim = raw_action_dim

    rng = np.random.default_rng(seed)
    if expected_action_dim == raw_action_dim:
        starts = _sample_valid_chunk_starts(
            episode_ids=episode_ids,
            step_idx=step_idx,
            seq_len=action_data.shape[0],
            chunk_len=chunk_len,
            num_chunks=num_chunks,
            rng=rng,
        )
        if starts.size == 0:
            return {
                "low": fallback_low,
                "high": fallback_high,
                "num_chunks": np.array(0, dtype=np.int64),
                "chunk_len": np.array(chunk_len, dtype=np.int64),
            }
        chunks = np.stack([action_data[s : s + chunk_len] for s in starts], axis=0)
    elif expected_action_dim > raw_action_dim and expected_action_dim % raw_action_dim == 0:
        # Model expects grouped actions (e.g., 10) while dataset stores primitive actions (e.g., 2).
        # Build chunks of `chunk_len` grouped tokens by concatenating contiguous primitive actions.
        group = expected_action_dim // raw_action_dim
        raw_chunk_len = chunk_len * group
        if action_data.shape[0] < raw_chunk_len:
            return {
                "low": fallback_low,
                "high": fallback_high,
                "num_chunks": np.array(0, dtype=np.int64),
                "chunk_len": np.array(chunk_len, dtype=np.int64),
            }
        starts = _sample_valid_chunk_starts(
            episode_ids=episode_ids,
            step_idx=step_idx,
            seq_len=action_data.shape[0],
            chunk_len=raw_chunk_len,
            num_chunks=num_chunks,
            rng=rng,
        )
        if starts.size == 0:
            return {
                "low": fallback_low,
                "high": fallback_high,
                "num_chunks": np.array(0, dtype=np.int64),
                "chunk_len": np.array(chunk_len, dtype=np.int64),
            }
        raw_chunks = np.stack([action_data[s : s + raw_chunk_len] for s in starts], axis=0)
        chunks = raw_chunks.reshape(num_chunks, chunk_len, expected_action_dim)
    else:
        warnings.warn(
            "Unable to match action dimension for latent-prior calibration: "
            f"dataset action dim={raw_action_dim}, model latent-action input dim={expected_action_dim}. "
            "Falling back to default latent bounds.",
            stacklevel=2,
        )
        return {
            "low": fallback_low,
            "high": fallback_high,
            "num_chunks": np.array(0, dtype=np.int64),
            "chunk_len": np.array(chunk_len, dtype=np.int64),
        }

    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    chunks_t = torch.from_numpy(chunks.astype(np.float32)).to(device)
    mask_t = torch.ones((chunks_t.size(0), chunks_t.size(1)), dtype=torch.bool, device=device)

    if not hasattr(model, "encode_macro_actions"):
        return {
            "low": fallback_low,
            "high": fallback_high,
            "num_chunks": np.array(0, dtype=np.int64),
            "chunk_len": np.array(chunk_len, dtype=np.int64),
        }

    latents = model.encode_macro_actions(chunks_t, mask_t).detach().cpu().numpy()
    if latents.ndim != 2 or latents.shape[0] < min_chunks:
        return {
            "low": fallback_low,
            "high": fallback_high,
            "num_chunks": np.array(latents.shape[0] if latents.ndim > 0 else 0, dtype=np.int64),
            "chunk_len": np.array(chunk_len, dtype=np.int64),
        }

    lo = np.percentile(latents, q_low, axis=0).astype(np.float32)
    hi = np.percentile(latents, q_high, axis=0).astype(np.float32)
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - margin_ratio * span
    hi = hi + margin_ratio * span

    clamp_abs = cfg.get("clamp_abs", None)
    if clamp_abs is not None:
        clamp_abs = float(clamp_abs)
        lo = np.clip(lo, -clamp_abs, clamp_abs)
        hi = np.clip(hi, -clamp_abs, clamp_abs)

    # ensure valid bounds
    bad = hi <= lo
    if np.any(bad):
        hi[bad] = lo[bad] + 1e-3

    return {
        "low": lo.astype(np.float32),
        "high": hi.astype(np.float32),
        "num_chunks": np.array(latents.shape[0], dtype=np.int64),
        "chunk_len": np.array(chunk_len, dtype=np.int64),
    }


class HierarchicalWorldModelPolicy:
    """Two-level MPC policy: high-level latent CEM + low-level primitive CEM."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        high_solver,
        low_solver,
        high_config,
        low_config,
        macro_replan_interval: int = 5,
        process: dict[str, Any] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        high_latent_bounds: dict[str, np.ndarray] | None = None,
    ):
        self.type = "hierarchical_world_model"
        self.env = None

        self.model = model.eval()
        self.high_solver = high_solver
        self.low_solver = low_solver
        self.high_cfg = high_config
        self.low_cfg = low_config
        self.macro_replan_interval = int(macro_replan_interval)
        self.process = process or {}
        self.transform = transform or {}
        self.high_latent_bounds = high_latent_bounds

        self._action_buffer: deque[torch.Tensor] = deque(maxlen=self.flatten_receding_horizon_low)
        self._next_low_init: torch.Tensor | None = None
        self._next_high_init: torch.Tensor | None = None
        self._z_subgoal: torch.Tensor | None = None
        self._steps_since_high: int = 10**9
        self._latent_action_dim = _infer_latent_dim_from_model(self.model)
        self._low_grouped_action_dim: int | None = None

    @property
    def flatten_receding_horizon_low(self) -> int:
        return int(self.low_cfg.receding_horizon) * int(self.low_cfg.action_block)

    def set_high_latent_bounds(self, bounds: dict[str, np.ndarray]) -> None:
        self.high_latent_bounds = bounds

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        """Apply preprocessors/transforms following WorldModelPolicy behavior."""
        for k, v in info_dict.items():
            is_numpy = isinstance(v, (np.ndarray, np.generic))

            if k in self.process:
                if not is_numpy:
                    raise ValueError(f"Expected numpy array for key '{k}', got {type(v)}")
                shape = v.shape
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])
                v = self.process[k].transform(v)
                v = v.reshape(shape)
                is_numpy = True

            if k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])

                if k.startswith("pixels") or k.startswith("goal"):
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))
                    else:
                        v = v.permute(0, 3, 1, 2)
                v = torch.stack([self.transform[k](tv_tensors.Image(x)) for x in v])

                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])
                is_numpy = isinstance(v, (np.ndarray, np.generic))

            if is_numpy and getattr(v, "dtype", None) is not None and v.dtype.kind not in "USO":
                v = torch.from_numpy(v)
            info_dict[k] = v

        return info_dict

    def _device(self) -> torch.device:
        for p in self.model.parameters():
            return p.device
        return torch.device("cpu")

    def _build_high_action_space(self, n_envs: int) -> Box:
        if self.high_latent_bounds is not None:
            low_1d = np.asarray(self.high_latent_bounds["low"], dtype=np.float32)
            high_1d = np.asarray(self.high_latent_bounds["high"], dtype=np.float32)
        else:
            low_1d = -np.ones((self._latent_action_dim,), dtype=np.float32)
            high_1d = np.ones((self._latent_action_dim,), dtype=np.float32)

        if low_1d.shape != high_1d.shape:
            raise ValueError("high_latent_bounds low/high must have the same shape")
        if low_1d.ndim != 1:
            raise ValueError("high_latent_bounds low/high must be 1D arrays")

        low = np.broadcast_to(low_1d[None, :], (n_envs, low_1d.shape[0])).copy()
        high = np.broadcast_to(high_1d[None, :], (n_envs, high_1d.shape[0])).copy()
        return Box(low=low, high=high, shape=(n_envs, low_1d.shape[0]), dtype=np.float32)

    def set_env(self, env: Any) -> None:
        self.env = env
        n_envs = int(getattr(env, "num_envs", 1))

        self.low_solver.configure(
            action_space=env.action_space,
            n_envs=n_envs,
            config=self.low_cfg,
        )
        self._low_grouped_action_dim = int(self.low_solver.action_dim)

        high_action_space = self._build_high_action_space(n_envs)
        self.high_solver.configure(
            action_space=high_action_space,
            n_envs=n_envs,
            config=self.high_cfg,
        )

        self._action_buffer = deque(maxlen=self.flatten_receding_horizon_low)
        self._next_low_init = None
        self._next_high_init = None
        self._z_subgoal = None
        self._steps_since_high = 10**9

    @torch.inference_mode()
    def _encode_pixels_last(self, pixels: torch.Tensor) -> torch.Tensor:
        # Expect (B, T, C, H, W) or (B, C, H, W)
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        if pixels.ndim != 5:
            raise ValueError(f"Unsupported pixel shape for encoding: {tuple(pixels.shape)}")
        batch = {"pixels": pixels}
        out = self.model.encode(batch, encode_actions=False)
        return out["emb"][:, -1]

    @torch.inference_mode()
    def _plan_high(self, *, z_init: torch.Tensor, z_goal: torch.Tensor) -> None:
        high_info = {
            "planner_level": "high",
            "z_init": z_init,
            "z_goal": z_goal,
        }
        # Keep warm-start tensors on CPU to avoid mixed-device issues in CEM internals.
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

        # Keep warm-start tensors on CPU to avoid mixed-device issues in CEM internals.
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

        # Convert grouped planner actions (n_env, receding_h, action_dim*block)
        # into env-step actions (n_env, receding_h*block, action_dim).
        plan = plan.reshape(self.env.num_envs, self.flatten_receding_horizon_low, -1)
        self._action_buffer.extend(plan.transpose(0, 1))

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        assert self.env is not None, "Environment not set for policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        info_dict = self._prepare_info(dict(info_dict))
        device = self._device()

        pixels = _to_torch(info_dict["pixels"], device)
        goal = _to_torch(info_dict["goal"], device)
        z_init = self._encode_pixels_last(pixels)
        z_goal = self._encode_pixels_last(goal)

        need_high = self._z_subgoal is None or self._steps_since_high >= self.macro_replan_interval
        if need_high:
            self._plan_high(z_init=z_init, z_goal=z_goal)

        if len(self._action_buffer) == 0:
            self._plan_low(z_init=z_init)

        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape)
        action_np = action.detach().cpu().numpy()

        if "action" in self.process:
            action_np = self.process["action"].inverse_transform(action_np)

        self._steps_since_high += 1
        return action_np



class StagedHierarchicalWorldModelPolicy(HierarchicalWorldModelPolicy):
    """One-shot high-level planner with stage-wise low-level tracking.

    This diagnostic policy solves the high-level plan exactly once at episode
    start, keeps the full predicted high-level rollout, and exposes those
    predicted latents sequentially as fixed stage targets for the low-level
    planner. The low-level planner still replans in closed loop.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        high_solver,
        low_solver,
        high_config,
        low_config,
        stage_duration_steps: int,
        clear_low_buffer_on_stage_change: bool = True,
        macro_replan_interval: int = 5,
        process: dict[str, Any] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        high_latent_bounds: dict[str, np.ndarray] | None = None,
    ):
        super().__init__(
            model=model,
            high_solver=high_solver,
            low_solver=low_solver,
            high_config=high_config,
            low_config=low_config,
            macro_replan_interval=macro_replan_interval,
            process=process,
            transform=transform,
            high_latent_bounds=high_latent_bounds,
        )
        if int(stage_duration_steps) <= 0:
            raise ValueError("stage_duration_steps must be > 0.")
        self.stage_duration_steps = int(stage_duration_steps)
        self.clear_low_buffer_on_stage_change = bool(clear_low_buffer_on_stage_change)
        self._stage_targets: torch.Tensor | None = None
        self._active_stage_idx: int = 0
        self._steps_total: int = 0

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        self._stage_targets = None
        self._active_stage_idx = 0
        self._steps_total = 0

    @torch.inference_mode()
    def _plan_high(self, *, z_init: torch.Tensor, z_goal: torch.Tensor) -> None:
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
        self._steps_since_high = 0
        self._set_active_stage(0)

    def _set_active_stage(self, stage_idx: int) -> None:
        if self._stage_targets is None:
            raise RuntimeError("Stage targets must be initialized before setting a stage.")

        clamped_idx = max(0, min(int(stage_idx), int(self._stage_targets.size(1)) - 1))
        stage_changed = self._z_subgoal is None or clamped_idx != self._active_stage_idx
        self._active_stage_idx = clamped_idx
        self._z_subgoal = self._stage_targets[:, clamped_idx, :]

        if stage_changed:
            self._next_low_init = None
            if self.clear_low_buffer_on_stage_change:
                self._action_buffer.clear()

    def _sync_active_stage(self) -> None:
        if self._stage_targets is None:
            raise RuntimeError("Stage targets must be initialized before syncing stages.")

        target_idx = min(
            self._steps_total // self.stage_duration_steps,
            int(self._stage_targets.size(1)) - 1,
        )
        self._set_active_stage(target_idx)

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        assert self.env is not None, "Environment not set for policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        info_dict = self._prepare_info(dict(info_dict))
        device = self._device()

        pixels = _to_torch(info_dict["pixels"], device)
        goal = _to_torch(info_dict["goal"], device)
        z_init = self._encode_pixels_last(pixels)
        z_goal = self._encode_pixels_last(goal)

        if self._stage_targets is None:
            self._plan_high(z_init=z_init, z_goal=z_goal)
        else:
            self._sync_active_stage()

        if len(self._action_buffer) == 0:
            self._plan_low(z_init=z_init)

        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape)
        action_np = action.detach().cpu().numpy()

        if "action" in self.process:
            action_np = self.process["action"].inverse_transform(action_np)

        self._steps_total += 1
        self._steps_since_high += 1
        return action_np
