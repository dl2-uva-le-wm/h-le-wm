from __future__ import annotations

import torch
from torch.utils.data import default_collate

from h_le_wm.models.waypoint_sampling import sample_waypoints


def gather_waypoint_embeddings(emb: torch.Tensor, waypoints: torch.Tensor) -> torch.Tensor:
    """Gather latent embeddings at sampled waypoint indices."""
    if emb.ndim != 3:
        raise ValueError("emb must be shape (B, T, D)")
    if waypoints.ndim != 2:
        raise ValueError("waypoints must be shape (B, N)")
    b = emb.size(0)
    batch_idx = torch.arange(b, device=emb.device).unsqueeze(1)
    return emb[batch_idx, waypoints]


def build_action_chunks(
    actions: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded variable-length action chunks and validity mask."""
    if actions.ndim != 3:
        raise ValueError("actions must be shape (B, T, A)")
    if starts.ndim != 1 or ends.ndim != 1:
        raise ValueError("starts/ends must be shape (B,)")
    if starts.shape != ends.shape:
        raise ValueError("starts and ends must have matching shape")

    lengths = (ends - starts).to(dtype=torch.long)
    if (lengths <= 0).any():
        raise ValueError("All action chunks must have positive length")

    b, _t, act_dim = actions.shape
    max_len = int(lengths.max().item())
    chunks = actions.new_zeros((b, max_len, act_dim))
    mask = torch.zeros((b, max_len), dtype=torch.bool, device=actions.device)

    for i in range(b):
        s = int(starts[i].item())
        e = int(ends[i].item())
        l = e - s
        chunks[i, :l] = actions[i, s:e]
        mask[i, :l] = True

    return chunks, mask


def build_action_chunks_batched(
    actions: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded action chunks for all waypoint transitions in one pass."""
    if actions.ndim != 3:
        raise ValueError("actions must be shape (B, T, A)")
    if starts.ndim != 2 or ends.ndim != 2:
        raise ValueError("starts/ends must be shape (B, K)")
    if starts.shape != ends.shape:
        raise ValueError("starts and ends must have matching shape")

    b, t, act_dim = actions.shape
    if starts.size(0) != b:
        raise ValueError("starts/ends batch dimension must match actions")
    if (starts < 0).any() or (ends < 0).any() or (starts >= t).any() or (ends > t).any():
        raise ValueError("starts/ends must satisfy 0 <= starts < T and 0 <= ends <= T")

    lengths = (ends - starts).to(dtype=torch.long)
    if (lengths <= 0).any():
        raise ValueError("All action chunks must have positive length")

    max_len = int(lengths.max().item())
    offsets = torch.arange(max_len, device=actions.device).view(1, 1, max_len)
    mask = offsets < lengths.unsqueeze(-1)

    gather_idx = starts.unsqueeze(-1) + offsets
    gather_idx = gather_idx.clamp(min=0, max=t - 1)

    batch_idx = torch.arange(b, device=actions.device).view(b, 1, 1)
    batch_idx = batch_idx.expand_as(gather_idx)
    chunks = actions[batch_idx, gather_idx, :]
    chunks = chunks * mask.unsqueeze(-1).to(dtype=actions.dtype)

    if chunks.shape[-1] != act_dim:
        raise RuntimeError("Unexpected chunk action dimension mismatch")
    return chunks, mask


def build_p2_frozen_waypoint_collate(cfg, pixel_preprocessor):
    """Build collate_fn that preprocesses only sampled waypoint pixel frames."""
    if pixel_preprocessor is None:
        raise ValueError("pixel_preprocessor is required for P2 frozen waypoint collate.")

    num_waypoints = int(cfg.wm.high_level.waypoints.num)

    def collate(samples):
        if len(samples) == 0:
            raise ValueError("Cannot collate an empty batch.")

        seq_len = int(samples[0]["action"].shape[0])
        waypoints, _ = sample_waypoints(
            cfg,
            batch_size=len(samples),
            seq_len=seq_len,
            device="cpu",
        )

        processed = []
        for i, sample in enumerate(samples):
            item = dict(sample)
            wp = waypoints[i]
            pixels = item["pixels"]
            if torch.is_tensor(pixels):
                selected = pixels.index_select(
                    0,
                    wp.to(device=pixels.device, dtype=torch.long),
                )
            else:
                selected = pixels[wp.cpu().numpy()]

            pixel_out = pixel_preprocessor({"pixels": selected})
            item["pixels"] = pixel_out["pixels"]
            item["waypoints"] = wp.clone()
            processed.append(item)

        batch = default_collate(processed)
        if batch["pixels"].shape[1] != num_waypoints:
            raise RuntimeError(
                f"Expected waypoint pixel length={num_waypoints}, got {batch['pixels'].shape[1]}"
            )
        return batch

    return collate
