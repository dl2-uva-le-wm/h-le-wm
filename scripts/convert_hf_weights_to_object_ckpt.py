#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.errors import InstantiationException
from omegaconf import OmegaConf


def _default_cache_dir() -> Path:
    env = os.environ.get("STABLEWM_HOME")
    if env:
        print(f"Using STABLEWM_HOME from environment: {env}")
        return Path(env).expanduser().resolve()
    
    print("STABLEWM_HOME not set, using default cache directory ~/.stable-wm")
    return Path("~/.stable-wm").expanduser().resolve()


def _load_config(config_path: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(str(config_path))
    cfg_obj = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_obj, dict):
        raise ValueError(f"Expected config object dict, got: {type(cfg_obj)}")
    if "_target_" not in cfg_obj:
        raise ValueError(
            f"{config_path} does not contain a hydra target (_target_). "
            "Cannot instantiate model object."
        )
    return cfg_obj


def _extract_state_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        # Common wrapper keys.
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            val = raw.get(key)
            if isinstance(val, dict):
                return val
        # Or the dict itself is already a state dict.
        return raw
    raise TypeError(f"Unsupported weights payload type: {type(raw)}")


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not prefix:
        return state_dict
    plen = len(prefix)
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        out[key[plen:] if key.startswith(prefix) else key] = value
    return out


def _parse_hf_url(url: str) -> tuple[str, str]:
    """
    Parse Hugging Face URL and return (repo_id, revision).

    Supported examples:
    - https://huggingface.co/quentinll/lewm-pusht
    - https://huggingface.co/quentinll/lewm-pusht/tree/main
    - https://huggingface.co/quentinll/lewm-pusht/tree/some-branch
    """
    m = re.match(r"^https?://huggingface\.co/([^/]+/[^/]+)(?:/tree/([^/?#]+))?", url)
    if not m:
        raise ValueError(f"Unsupported Hugging Face URL format: {url}")
    repo_id = m.group(1)
    revision = m.group(2) or "main"
    return repo_id, revision


def _download_file(url: str, out_path: Path) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "lewm-hf-converter/1.0",
        },
    )
    with urllib.request.urlopen(req) as r:  # nosec B310 - trusted user URL + HF resolve endpoints
        data = r.read()
    out_path.write_bytes(data)


def _download_hf_pair(hf_url: str) -> tuple[Path, Path, str]:
    repo_id, revision = _parse_hf_url(hf_url)
    tmp_dir = Path(tempfile.mkdtemp(prefix="lewm_hf_convert_"))
    config_path = tmp_dir / "config.json"
    weights_path = tmp_dir / "weights.pt"

    base = f"https://huggingface.co/{repo_id}/resolve/{revision}"
    config_url = f"{base}/config.json?download=true"
    weights_url = f"{base}/weights.pt?download=true"

    _download_file(config_url, config_path)
    _download_file(weights_url, weights_path)
    return weights_path, config_path, f"{repo_id}@{revision}"


def _instantiate_model_with_compat(cfg_obj: dict[str, Any]) -> torch.nn.Module:
    """
    Instantiate model from config with backward-compat fallback.

    Some environments ship a stable_worldmodel version that does not expose
    stable_worldmodel.wm.lewm.LeWM yet. In that case, retry with local
    third_party/lewm implementation.
    """
    try:
        model = hydra.utils.instantiate(cfg_obj)
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Instantiated object is not a torch module: {type(model)}")
        return model
    except InstantiationException as exc:
        target = str(cfg_obj.get("_target_", ""))
        needs_fallback = (
            "stable_worldmodel.wm.lewm.LeWM" in str(exc)
            or target == "stable_worldmodel.wm.lewm.LeWM"
        )
        if not needs_fallback:
            raise

        repo_root = Path(__file__).resolve().parents[1]
        local_lewm_dir = repo_root / "third_party" / "lewm"
        if not local_lewm_dir.exists():
            raise RuntimeError(
                "Compatibility fallback failed: third_party/lewm not found."
            ) from exc

        if str(local_lewm_dir) not in sys.path:
            sys.path.insert(0, str(local_lewm_dir))

        target_map = {
            "stable_worldmodel.wm.lewm.LeWM": "jepa.JEPA",
            "stable_worldmodel.wm.lewm.module.Predictor": "module.ARPredictor",
            "stable_worldmodel.wm.lewm.module.ARPredictor": "module.ARPredictor",
            "stable_worldmodel.wm.lewm.module.Embedder": "module.Embedder",
            "stable_worldmodel.wm.lewm.module.MLP": "module.MLP",
            "stable_worldmodel.wm.lewm.module.SIGReg": "module.SIGReg",
            "stable_worldmodel.wm.lewm.module.Transformer": "module.Transformer",
            "stable_worldmodel.wm.lewm.module.ConditionalBlock": "module.ConditionalBlock",
        }

        def rewrite_targets(x: Any) -> Any:
            if isinstance(x, dict):
                y: dict[str, Any] = {}
                for k, v in x.items():
                    if k == "_target_" and isinstance(v, str):
                        y[k] = target_map.get(v, v)
                    else:
                        y[k] = rewrite_targets(v)
                return y
            if isinstance(x, list):
                return [rewrite_targets(v) for v in x]
            return x

        cfg_fallback = rewrite_targets(cfg_obj)
        model = hydra.utils.instantiate(cfg_fallback)
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Fallback instantiated object is not a torch module: {type(model)}")
        print(
            "Compat mode: mapped _target_ stable_worldmodel.wm.lewm.LeWM -> jepa.JEPA "
            f"from {local_lewm_dir}"
        )
        return model


def _try_load_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
    *,
    strict: bool,
) -> tuple[bool, str]:
    attempts = [
        ("as-is", state_dict),
        ("strip module.", _strip_prefix(state_dict, "module.")),
        ("strip model.", _strip_prefix(state_dict, "model.")),
        ("strip model.module.", _strip_prefix(_strip_prefix(state_dict, "model."), "module.")),
    ]
    errors: list[str] = []
    for name, candidate in attempts:
        try:
            result = model.load_state_dict(candidate, strict=strict)
            missing = list(getattr(result, "missing_keys", []))
            unexpected = list(getattr(result, "unexpected_keys", []))
            msg = (
                f"Loaded with strategy={name}, strict={strict}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            return True, msg
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    return False, "\n".join(errors)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert HuggingFace-style LEWM weights (weights.pt + config.json) "
            "into stable-worldmodel object checkpoint format (*_object.ckpt)."
        )
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--hf-url",
        help=(
            "Hugging Face model page URL, for example "
            "https://huggingface.co/quentinll/lewm-pusht/tree/main"
        ),
    )
    src_group.add_argument("--weights", type=Path, help="Path to local weights.pt")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to local config.json (required when --weights is used)",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help=(
            "StableWM run name relative to cache dir (example: pusht/lewm). "
            "Output will be <cache-dir>/<run-name>_object.ckpt"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override STABLEWM_HOME (default: $STABLEWM_HOME or ~/.stable-wm)",
    )
    parser.add_argument(
        "--allow-non-strict",
        action="store_true",
        help="Fallback to non-strict state dict load if strict load fails.",
    )
    args = parser.parse_args()

    if args.hf_url:
        weights_path, config_path, src_desc = _download_hf_pair(args.hf_url)
        print(f"Downloaded HF artifacts from {src_desc}")
    else:
        if args.config is None:
            raise ValueError("--config is required when using --weights")
        weights_path = args.weights.expanduser().resolve()
        config_path = args.config.expanduser().resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"weights not found: {weights_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"config not found: {config_path}")

    cache_dir = (args.cache_dir.expanduser().resolve() if args.cache_dir else _default_cache_dir())
    out_path = (cache_dir / f"{args.run_name}_object.ckpt").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_obj = _load_config(config_path)
    try:
        model = _instantiate_model_with_compat(cfg_obj)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not instantiate model from config. "
            "Your stable-worldmodel version may be outdated.\n"
            "Try: uv pip install -U stable-worldmodel[train,env]\n"
            f"Original error: {exc}"
        ) from exc

    raw = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(raw)

    ok, message = _try_load_state_dict(model, state_dict, strict=True)
    if not ok and args.allow_non_strict:
        ok, message = _try_load_state_dict(model, state_dict, strict=False)
    if not ok:
        raise RuntimeError(
            "Failed to load weights into model.\n"
            "Try --allow-non-strict if you are sure config/weights match.\n"
            f"{message}"
        )

    model.eval()
    model.requires_grad_(False)
    torch.save(model, out_path)

    print(message)
    print(f"Saved object checkpoint: {out_path}")
    print(
        "Use this in eval (policy path omits _object.ckpt suffix): "
        f"policy={args.run_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
