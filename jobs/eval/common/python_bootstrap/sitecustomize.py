from __future__ import annotations

import os

from h_le_wm.eval.determinism import DEFAULT_SEED, configure_process_determinism


if "EVAL_DETERMINISM" in os.environ or "EVAL_SEED" in os.environ:
    try:
        configure_process_determinism(
            seed=int(os.environ.get("EVAL_SEED", DEFAULT_SEED)),
            mode=os.environ.get("EVAL_DETERMINISM"),
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy", "torch"}:
            raise
