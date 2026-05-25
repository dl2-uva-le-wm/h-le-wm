from __future__ import annotations

import os
import random
from typing import Any



DEFAULT_DETERMINISM_MODE = "strict"
DEFAULT_SEED = 42


def _normalize_mode(mode: str | None) -> str:
    normalized = (mode or DEFAULT_DETERMINISM_MODE).strip().lower()
    if normalized != "strict":
        raise ValueError(
            f"Unsupported eval determinism mode '{mode}'. Only 'strict' is supported."
        )
    return normalized


def _set_default_env(seed: int, mode: str) -> None:
    os.environ["EVAL_SEED"] = str(seed)
    os.environ["EVAL_DETERMINISM"] = mode
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def configure_process_determinism(
    seed: int | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    resolved_seed = int(
        DEFAULT_SEED if seed is None else seed
    )
    resolved_mode = _normalize_mode(mode or os.environ.get("EVAL_DETERMINISM"))
    _set_default_env(resolved_seed, resolved_mode)

    import numpy as np
    import torch

    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed(resolved_seed)
        torch.cuda.manual_seed_all(resolved_seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    return {
        "seed": resolved_seed,
        "mode": resolved_mode,
        "cuda_available": cuda_available,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False))
        if hasattr(torch.backends.cuda, "matmul")
        else None,
        "cudnn_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False))
        if hasattr(torch.backends.cudnn, "allow_tf32")
        else None,
    }


def format_determinism_report(report: dict[str, Any]) -> str:
    parts = [
        f"seed={report['seed']}",
        f"mode={report['mode']}",
        f"cuda_available={report['cuda_available']}",
        f"pythonhashseed={report['pythonhashseed']}",
        f"cublas_workspace_config={report['cublas_workspace_config']}",
        f"torch_deterministic_algorithms={report['torch_deterministic_algorithms']}",
        f"cudnn_deterministic={report['cudnn_deterministic']}",
        f"cudnn_benchmark={report['cudnn_benchmark']}",
    ]
    if report["cuda_tf32"] is not None:
        parts.append(f"cuda_tf32={report['cuda_tf32']}")
    if report["cudnn_tf32"] is not None:
        parts.append(f"cudnn_tf32={report['cudnn_tf32']}")
    return "[determinism] " + ", ".join(parts)
