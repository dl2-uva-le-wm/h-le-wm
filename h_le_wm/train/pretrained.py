from __future__ import annotations

import re
import sys
from pathlib import Path

import stable_worldmodel as swm
import torch

from h_le_wm.baseline.adapter import BASELINE_ROOT


def _object_epoch(path: Path, source_policy: str) -> int | None:
    """Extract epoch index from an object-checkpoint filename."""
    match = re.match(rf"^{re.escape(source_policy)}_epoch_(\d+)_object\.ckpt$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def resolve_pretrained_checkpoint(cfg) -> Path:
    """Resolve pretrained low-level object checkpoint according to Hydra config."""
    pcfg = cfg.pretrained_low_level
    cpcfg = pcfg.checkpoint

    explicit = cpcfg.get("path")
    if explicit not in (None, ""):
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Explicit pretrained checkpoint does not exist: {path}")
        return path

    mode = str(cpcfg.get("selection_mode", "latest"))
    if mode == "explicit_path":
        raise ValueError(
            "pretrained_low_level.checkpoint.selection_mode=explicit_path requires "
            "pretrained_low_level.checkpoint.path to be set."
        )

    source_policy = str(pcfg.get("source_policy", "")).strip()
    if not source_policy:
        raise ValueError("pretrained_low_level.source_policy must be provided.")

    search_dir_raw = cpcfg.get("search_dir")
    search_dir = Path(search_dir_raw).expanduser() if search_dir_raw else Path(
        swm.data.utils.get_cache_dir()
    )
    if not search_dir.exists():
        raise FileNotFoundError(f"Checkpoint search_dir does not exist: {search_dir}")

    if mode == "epoch":
        epoch = int(cpcfg.get("epoch", 0))
        if epoch <= 0:
            raise ValueError("checkpoint.epoch must be > 0 when selection_mode=epoch")
        path = search_dir / f"{source_policy}_epoch_{epoch}_object.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"Epoch checkpoint not found: {path}")
        return path

    if mode not in {"latest", "best"}:
        raise ValueError(
            f"Unsupported pretrained checkpoint selection_mode={mode}. "
            "Use one of: latest, best, epoch, explicit_path."
        )

    if mode == "best":
        best_path = search_dir / f"{source_policy}_best_object.ckpt"
        if best_path.exists():
            return best_path

    candidates = []
    for path in search_dir.glob(f"{source_policy}_epoch_*_object.ckpt"):
        epoch = _object_epoch(path, source_policy)
        if epoch is not None:
            candidates.append((epoch, path))

    if not candidates:
        raise FileNotFoundError(
            f"No object checkpoints found for source_policy='{source_policy}' in {search_dir}"
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_pretrained_low_level_model(path: Path):
    """Load baseline LEWM object checkpoint and return the JEPA model object."""
    baseline_root = str(BASELINE_ROOT)
    if baseline_root not in sys.path:
        sys.path.insert(0, baseline_root)

    try:
        model_obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        model_obj = torch.load(path, map_location="cpu")

    model = model_obj.model if hasattr(model_obj, "model") else model_obj
    required = ("encoder", "predictor", "action_encoder", "projector", "pred_proj")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(
            f"Loaded object checkpoint does not look like LEWM JEPA model. Missing attrs: {missing}"
        )
    return model
