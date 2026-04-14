from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent
BASELINE_ROOT = REPO_ROOT / "third_party" / "lewm"


def ensure_baseline_available() -> Path:
    if not BASELINE_ROOT.exists():
        raise RuntimeError(
            "Baseline submodule not found at third_party/lewm. "
            "Run: git submodule update --init --recursive"
        )
    return BASELINE_ROOT


_LOADED: Dict[Tuple[str, str], ModuleType] = {}


def _load_baseline_module(module_name: str, relative_path: str) -> ModuleType:
    key = (module_name, relative_path)
    if key in _LOADED:
        return _LOADED[key]

    root = ensure_baseline_available()
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Baseline module path not found: {path}")

    dynamic_name = f"_baseline_lewm_{module_name}"
    spec = importlib.util.spec_from_file_location(dynamic_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load baseline module from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[dynamic_name] = module
    old_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_flag
    _LOADED[key] = module
    return module


def baseline_script_path(script_name: str) -> Path:
    root = ensure_baseline_available()
    path = root / script_name
    if not path.exists():
        raise FileNotFoundError(f"Baseline script not found: {path}")
    return path


def run_baseline_script(script_name: str, args: list[str], *, extra_env: dict[str, str] | None = None) -> int:
    script = baseline_script_path(script_name)
    proc_env = None
    if extra_env:
        proc_env = dict(os.environ)
        proc_env.update(extra_env)

    cmd = [sys.executable, str(script), *args]
    if os.getenv("LEWM_WRAPPER_DRY_RUN") == "1":
        print("[dry-run] baseline wrapper:", " ".join(cmd))
        return 0
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=proc_env)


_EXPORTS = {
    # module.py
    "ARPredictor": ("module", "module.py", "ARPredictor"),
    "Embedder": ("module", "module.py", "Embedder"),
    "MLP": ("module", "module.py", "MLP"),
    "SIGReg": ("module", "module.py", "SIGReg"),
    "Transformer": ("module", "module.py", "Transformer"),
    "ConditionalBlock": ("module", "module.py", "ConditionalBlock"),
    # utils.py
    "ModelObjectCallBack": ("utils", "utils.py", "ModelObjectCallBack"),
    "get_column_normalizer": ("utils", "utils.py", "get_column_normalizer"),
    "get_img_preprocessor": ("utils", "utils.py", "get_img_preprocessor"),
}

__all__ = [
    "BASELINE_ROOT",
    "REPO_ROOT",
    "ensure_baseline_available",
    "baseline_script_path",
    "run_baseline_script",
    *sorted(_EXPORTS.keys()),
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, relpath, attr = _EXPORTS[name]
    module = _load_baseline_module(module_name, relpath)
    value = getattr(module, attr)
    globals()[name] = value
    return value
