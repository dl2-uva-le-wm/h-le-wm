from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import io
import re
import zipfile


PARAM_ORDER = [
    "high_horizon",
    "high_replan_interval",
    "high_action_block",
    "high_num_samples",
    "high_n_steps",
    "high_topk",
    "low_horizon",
    "low_action_block",
    "low_num_samples",
    "low_n_steps",
    "low_topk",
    "eval_budget",
]

ENV_MAP = {
    "GOAL_OFFSET_STEPS": "goal_offset_steps",
    "EVAL_BUDGET": "eval_budget",
    "RUN_NAME": "run_name",
    "CHECKPOINT_EPOCH": "checkpoint_epoch",
    "HIGH_HORIZON": "high_horizon",
    "HIGH_REPLAN_INTERVAL": "high_replan_interval",
    "HIGH_ACTION_BLOCK": "high_action_block",
    "HIGH_NUM_SAMPLES": "high_num_samples",
    "HIGH_N_STEPS": "high_n_steps",
    "HIGH_TOPK": "high_topk",
    "LOW_HORIZON": "low_horizon",
    "LOW_ACTION_BLOCK": "low_action_block",
    "LOW_NUM_SAMPLES": "low_num_samples",
    "LOW_N_STEPS": "low_n_steps",
    "LOW_TOPK": "low_topk",
}

DEFAULT_RE = re.compile(r'^(?:export\s+)?([A-Z0-9_]+)="\$\{\1:-([^}]*)\}"', re.MULTILINE)
LOWER_RE = re.compile(r'^LOWERH_SCRIPT="([^"]+)"', re.MULTILINE)
RESULT_ROW_RE = re.compile(r"^\| (?P<row>.+) \|$", re.MULTILINE)
PERCENT_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)%")


@dataclass(frozen=True)
class SourceFile:
    source_kind: str
    path: str
    text: str


def _repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "hi_eval.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root containing hi_eval.py")


def load_source_files(repo_root: Path | None = None) -> dict[str, SourceFile]:
    root = _repo_root(repo_root)
    files: dict[str, SourceFile] = {}

    zip_path = root / "tests" / "data" / "jobs_github_ready_20260422.zip"
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith((".sh", ".md")):
                continue
            if not name.startswith("jobs/eval/hi/"):
                continue
            files[name] = SourceFile(
                source_kind="zip",
                path=name,
                text=zf.read(name).decode("utf-8", errors="ignore"),
            )

    for path in sorted((root / "jobs" / "eval" / "hi").rglob("*.sh")):
        rel = path.relative_to(root).as_posix()
        files[rel] = SourceFile(source_kind="workspace", path=rel, text=path.read_text())

    return files


def _normalize_lower_ref(script_path: str, raw_ref: str) -> str | None:
    ref = raw_ref.replace("${SCRIPT_DIR}/", "")
    if "${" in ref:
        return None
    if not ref.endswith(".sh"):
        return None
    return str(PurePosixPath(script_path).parent.joinpath(ref).as_posix())


def _extract_defaults(text: str) -> dict[str, str]:
    defaults = {}
    for env_name, raw_value in DEFAULT_RE.findall(text):
        defaults[env_name] = raw_value
    return defaults


def _extract_parent_path(script_path: str, text: str) -> str | None:
    match = LOWER_RE.search(text)
    if not match:
        return None
    return _normalize_lower_ref(script_path, match.group(1))


def _coerce(value: str | None):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def build_script_catalog(repo_root: Path | None = None) -> list[dict]:
    files = load_source_files(repo_root)
    cache: dict[str, dict[str, str]] = {}

    def resolve(script_path: str, stack: tuple[str, ...] = ()) -> dict[str, str]:
        if script_path in cache:
            return cache[script_path]
        if script_path in stack or script_path not in files:
            cache[script_path] = {}
            return cache[script_path]

        source = files[script_path]
        merged: dict[str, str] = {}
        parent_path = _extract_parent_path(script_path, source.text)
        if parent_path:
            merged.update(resolve(parent_path, stack + (script_path,)))
        merged.update(_extract_defaults(source.text))
        cache[script_path] = merged
        return merged

    rows = []
    for path, source in sorted(files.items()):
        if not path.endswith(".sh"):
            continue
        if "/d25/" not in path and "/d50/" not in path:
            continue
        defaults = resolve(path)
        row = {
            "source_kind": source.source_kind,
            "script_path": path,
            "script_name": PurePosixPath(path).name,
            "goal_family": "d25" if "/d25/" in path else "d50",
            "parent_script": _extract_parent_path(path, source.text),
        }
        for env_name, col_name in ENV_MAP.items():
            row[col_name] = _coerce(defaults.get(env_name))
        rows.append(row)

    return sorted(rows, key=lambda row: (row.get("goal_offset_steps") or 0, row["script_path"]))


def _planner_to_params(text: str, prefix: str) -> dict[str, int | None]:
    if text == "n/a":
        return {
            f"{prefix}_horizon": None,
            f"{prefix}_action_block": None,
            f"{prefix}_num_samples": None,
            f"{prefix}_n_steps": None,
            f"{prefix}_topk": None,
            f"{prefix}_replan_interval": None,
        }

    out = {
        f"{prefix}_horizon": None,
        f"{prefix}_action_block": None,
        f"{prefix}_num_samples": None,
        f"{prefix}_n_steps": None,
        f"{prefix}_topk": None,
        f"{prefix}_replan_interval": None,
    }
    for raw_key, raw_val in re.findall(r"([a-z]+)=([0-9]+)", text):
        key = {
            "h": "horizon",
            "blk": "action_block",
            "samp": "num_samples",
            "iters": "n_steps",
            "topk": "topk",
            "rep": "replan_interval",
        }.get(raw_key)
        if key:
            out[f"{prefix}_{key}"] = int(raw_val)
    return out


def build_historical_results(repo_root: Path | None = None) -> list[dict]:
    files = load_source_files(repo_root)
    md_path = "jobs/eval/hi/PLANNING_HPARAM_RESULTS.md"
    if md_path not in files:
        raise FileNotFoundError(f"Missing {md_path}")
    text = files[md_path].text

    rows = []
    for line in io.StringIO(text):
        if not line.startswith("| `") or "Slurm out file" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 8:
            continue
        slurm_out, ckpt, mode, d_val, budget, high_planner, low_planner, success_rate = parts
        success_match = PERCENT_RE.search(success_rate)
        row = {
            "slurm_out_file": slurm_out.strip("`"),
            "checkpoint": ckpt.strip("`"),
            "mode": mode,
            "goal_offset_steps": int(d_val),
            "eval_budget": int(budget),
            "success_rate_label": success_rate,
            "success_rate_pct": float(success_match.group("value")) if success_match else None,
            "completed": success_match is not None,
        }
        row.update(_planner_to_params(high_planner.strip("`"), "high"))
        row.update(_planner_to_params(low_planner.strip("`"), "low"))
        rows.append(row)

    return sorted(
        rows,
        key=lambda row: (
            row["goal_offset_steps"],
            -(row["success_rate_pct"] if row["success_rate_pct"] is not None else -1),
        ),
    )


def build_analysis_frames(repo_root: Path | None = None):
    import pandas as pd

    scripts = build_script_catalog(repo_root)
    results = build_historical_results(repo_root)
    scripts_df = pd.DataFrame(scripts)
    results_df = pd.DataFrame(results)

    join_cols = ["goal_offset_steps", *PARAM_ORDER]
    matched = results_df.merge(
        scripts_df,
        how="left",
        on=join_cols,
        suffixes=("_result", "_script"),
    )
    return scripts_df, results_df, matched


def build_analysis_rows(repo_root: Path | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    scripts = build_script_catalog(repo_root)
    results = build_historical_results(repo_root)

    script_index: dict[tuple, list[dict]] = {}
    for script in scripts:
        key = tuple(script.get(col) for col in ["goal_offset_steps", *PARAM_ORDER])
        script_index.setdefault(key, []).append(script)

    matched_rows = []
    for result in results:
        key = tuple(result.get(col) for col in ["goal_offset_steps", *PARAM_ORDER])
        candidates = script_index.get(key) or [None]
        for script in candidates:
            row = dict(result)
            if script:
                row.update(
                    {
                        "script_path": script.get("script_path"),
                        "script_name": script.get("script_name"),
                        "source_kind": script.get("source_kind"),
                    }
                )
            else:
                row.update({"script_path": None, "script_name": None, "source_kind": None})
            matched_rows.append(row)

    return scripts, results, matched_rows
