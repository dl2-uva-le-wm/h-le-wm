from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from h_le_wm.paths import (
    REPO_ROOT,
    build_path_context,
    resolve_output_path,
    runs_root,
    spec_output_root,
    stablewm_home,
)


SUCCESS_RE = re.compile(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)")
INDEX_PATH = REPO_ROOT / "h_le_wm" / "experiments" / "index.yaml"
SPECS_ROOT = REPO_ROOT / "h_le_wm" / "experiments" / "specs"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(data)}")
    return data


def load_index_entries() -> list[dict[str, str]]:
    data = load_yaml(INDEX_PATH)
    specs = data.get("specs", [])
    if not isinstance(specs, list):
        raise ValueError("index.yaml must define a list under 'specs'")

    required_fields = ("name", "kind", "operation", "path")
    entries: list[dict[str, str]] = []
    for item in specs:
        if not isinstance(item, dict):
            raise ValueError("Each specs entry must be a mapping")
        entry: dict[str, str] = {}
        for field in required_fields:
            value = str(item.get(field, "")).strip()
            if not value:
                raise ValueError(f"Each specs entry must define '{field}'")
            entry[field] = value
        entries.append(entry)
    return entries


def load_index() -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in load_index_entries():
        out[entry["name"]] = entry["path"]
    return out


def resolve_spec_path(spec_ref: str) -> Path:
    candidate = Path(spec_ref)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        repo_candidate = (REPO_ROOT / candidate).resolve()
        if repo_candidate.is_file():
            return repo_candidate

    index = load_index()
    if spec_ref not in index:
        raise KeyError(f"Unknown spec '{spec_ref}'. See {INDEX_PATH}.")
    return (REPO_ROOT / index[spec_ref]).resolve()


def spec_slug(spec_name: str) -> str:
    return spec_name.replace("/", "__")


def output_root_for_spec(spec: dict[str, Any], context: dict[str, Any]) -> Path | None:
    outputs = spec.get("outputs", {})
    if not isinstance(outputs, dict) or not outputs:
        return None

    root_kind = str(outputs.get("root_kind", "repro")).strip() or "repro"
    root_name_template = str(outputs.get("root_name", "{spec_slug}")).strip() or "{spec_slug}"
    root_name = str(format_value(root_name_template, context))
    return spec_output_root(root_kind=root_kind, spec_slug=str(context["spec_slug"]), root_name=root_name)


def context_for_spec(spec: dict[str, Any]) -> dict[str, Any]:
    name = str(spec.get("name", "")).strip()
    context: dict[str, Any] = {
        **build_path_context(),
        "spec_name": name,
        "spec_slug": spec_slug(name),
    }
    output_root = output_root_for_spec(spec, context)
    if output_root is not None:
        context["output_root"] = output_root
    extra = spec.get("context", {})
    if isinstance(extra, dict):
        for key, value in extra.items():
            context[key] = format_value(value, context)
    return context


def format_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [format_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: format_value(item, context) for key, item in value.items()}
    return value


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def completion_path(spec: dict[str, Any], context: dict[str, Any]) -> Path | None:
    outputs = spec.get("outputs", {})
    if not isinstance(outputs, dict):
        return None
    raw = outputs.get("completion_file")
    if not raw:
        return None
    return resolve_output_path(str(raw), context=context)


def write_completion_file(spec: dict[str, Any], context: dict[str, Any]) -> None:
    done = completion_path(spec, context)
    if done is None:
        return
    ensure_parent(done)
    done.write_text(f"{spec.get('name', '<unnamed>')}\n")


def command_env(spec: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env_block = spec.get("env", {})
    if not isinstance(env_block, dict):
        return env
    for key, value in format_value(env_block, context).items():
        env[str(key)] = str(value)
    return env


def normalize_command(argv: list[str]) -> list[str]:
    if not argv:
        raise ValueError("command.argv must not be empty")
    out = list(argv)
    if out[0] == "python":
        out[0] = sys.executable
    return out


def run_subprocess(
    argv: list[str],
    *,
    env: dict[str, str],
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    dry_run: bool,
) -> int:
    pretty = " ".join(shlex_quote(part) for part in argv)
    print(pretty)
    if dry_run:
        return 0

    stdout_handle = None
    stderr_handle = None
    try:
        if stdout_path is not None:
            ensure_parent(stdout_path)
            stdout_handle = stdout_path.open("w")
        if stderr_path is not None:
            ensure_parent(stderr_path)
            stderr_handle = stderr_path.open("w")
        proc = subprocess.run(
            argv,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            stdout=stdout_handle or subprocess.PIPE,
            stderr=stderr_handle or subprocess.PIPE,
            check=False,
        )
        if stdout_handle is None and proc.stdout:
            sys.stdout.write(proc.stdout)
        if stderr_handle is None and proc.stderr:
            sys.stderr.write(proc.stderr)
        return proc.returncode
    finally:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()


def shlex_quote(part: str) -> str:
    if not part or any(ch in part for ch in " \t\n\"'`$"):
        return repr(part)
    return part


def run_command_spec(spec: dict[str, Any], *, force: bool, dry_run: bool) -> None:
    context = context_for_spec(spec)
    done = completion_path(spec, context)
    if done is not None and done.exists() and not force:
        print(f"[skip] completion file exists: {done}")
        return

    command = spec.get("command", {})
    if not isinstance(command, dict):
        raise ValueError("command spec must define a mapping under 'command'")
    argv = normalize_command(format_value(command.get("argv", []), context))
    env = command_env(spec, context)
    rc = run_subprocess(argv, env=env, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"Command spec '{spec.get('name')}' failed with exit code {rc}")
    if not dry_run:
        write_completion_file(spec, context)


def read_matrix_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        return [{key.strip(): (value or "").strip() for key, value in row.items()} for row in reader]


def parse_success_rate(text: str) -> str:
    matches = SUCCESS_RE.findall(text)
    return matches[-1] if matches else ""


def read_text_if_exists(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def build_baseline_matrix_command(
    *,
    spec: dict[str, Any],
    checkpoint: dict[str, Any],
    row: dict[str, str],
    row_index: int,
    seed: int,
    context: dict[str, Any],
) -> tuple[list[str], Path, Path, dict[str, str]]:
    policy = str(checkpoint["policy"])
    config_name = str(spec["config_name"])
    result_filename = str(spec.get("result_filename", "baseline_results.txt"))
    root_dir = Path(context["output_root"]) / checkpoint["slug"] / f"seed_{seed:03d}" / f"row_{row_index:03d}"
    result_path = root_dir / result_filename
    manifest_path = result_path.with_name(f"{result_path.stem}_episodes.tsv")
    argv = [
        sys.executable,
        "-m",
        "h_le_wm.eval.baseline_manifest",
        f"--config-name={config_name}",
        f"policy={policy}",
        f"seed={seed}",
        f"eval.dataset_name={spec['dataset_name']}",
        f"eval.num_eval={spec['num_eval']}",
        f"eval.goal_offset_steps={row['goal_offset'][-2:]}",
        f"eval.eval_budget={row['eval_budget']}",
        f"+eval.device={spec['eval_device']}",
        f"plan_config.horizon={row['low_horizon']}",
        f"plan_config.receding_horizon={row['low_receding_horizon']}",
        f"plan_config.action_block={row['low_action_block']}",
        f"solver.num_samples={row['low_num_samples']}",
        f"solver.n_steps={row['low_n_steps']}",
        f"solver.device={spec['eval_device']}",
        f"output.filename={result_filename}",
        f"+output.root_dir={root_dir}",
    ]
    row_meta = {
        "backend": "baseline",
        "policy": policy,
        "config_name": config_name,
        "checkpoint_slug": checkpoint["slug"],
        "checkpoint_path": str(stablewm_home() / checkpoint["object_relpath"]),
        "goal_offset": row["goal_offset"],
        "goal_offset_steps": row["goal_offset"][-2:],
        "eval_budget": row["eval_budget"],
        "low_horizon": row["low_horizon"],
        "low_receding_horizon": row["low_receding_horizon"],
        "low_action_block": row["low_action_block"],
        "low_num_samples": row["low_num_samples"],
        "low_n_steps": row["low_n_steps"],
        "seed": str(seed),
        "result_file": str(result_path),
        "episode_manifest": str(manifest_path),
    }
    return argv, result_path, manifest_path, row_meta


def normalize_planning_mode(config_note: str) -> str:
    note = (config_note or "").strip()
    if not note:
        return "hierarchical"
    if note == "hiStaged":
        return "hierarchical_staged"
    raise ValueError(f"Unsupported config_note '{note}'")


def build_hierarchical_matrix_command(
    *,
    spec: dict[str, Any],
    checkpoint: dict[str, Any],
    row: dict[str, str],
    row_index: int,
    seed: int,
    context: dict[str, Any],
) -> tuple[list[str], Path, Path, dict[str, str]]:
    run_name = str(checkpoint["run_name"])
    epoch = str(checkpoint["checkpoint_epoch"])
    policy = f"runs/{run_name}/{run_name}_epoch_{epoch}"
    result_filename = str(spec.get("result_filename", "hi_results.txt"))
    root_dir = Path(context["output_root"]) / checkpoint["slug"] / f"seed_{seed:03d}" / f"row_{row_index:03d}"
    result_path = root_dir / result_filename
    manifest_path = result_path.with_name(f"{result_path.stem}_episodes.tsv")
    argv = [
        sys.executable,
        "-m",
        "h_le_wm.eval.hierarchical",
        f"--config-name={spec['config_name']}",
        f"policy={policy}",
        f"seed={seed}",
        f"planning.mode={normalize_planning_mode(row.get('config_note', ''))}",
        f"eval.num_eval={spec['num_eval']}",
        f"eval.goal_offset_steps={row['goal_offset'][-2:]}",
        f"eval.eval_budget={row['eval_budget']}",
        f"planning.high.solver.device={spec['eval_device']}",
        f"planning.low.solver.device={spec['eval_device']}",
        f"solver.device={spec['eval_device']}",
        f"planning.high.solver.num_samples={row['high_num_samples']}",
        f"planning.high.solver.n_steps={row['high_n_steps']}",
        f"planning.high.solver.topk={row['high_topk']}",
        f"planning.high.plan_config.horizon={row['high_horizon']}",
        f"planning.high.plan_config.receding_horizon={row['high_receding_horizon']}",
        f"planning.high.plan_config.action_block={row['high_action_block']}",
        f"planning.high.replan_interval={row['high_replan_interval']}",
        f"planning.low.solver.num_samples={row['low_num_samples']}",
        f"planning.low.solver.n_steps={row['low_n_steps']}",
        f"planning.low.solver.topk={row['low_topk']}",
        f"planning.low.plan_config.horizon={row['low_horizon']}",
        f"planning.low.plan_config.receding_horizon={row['low_receding_horizon']}",
        f"planning.low.plan_config.action_block={row['low_action_block']}",
        f"output.filename={result_filename}",
        f"+output.root_dir={root_dir}",
    ]
    empirical = spec.get("empirical_macro", {})
    if isinstance(empirical, dict):
        for key, value in empirical.items():
            argv.append(f"planning.high.empirical_macro.{key}={value}")
    row_meta = {
        "backend": "hierarchical",
        "run_name": run_name,
        "checkpoint_epoch": epoch,
        "policy": policy,
        "config_name": spec["config_name"],
        "checkpoint_slug": checkpoint["slug"],
        "checkpoint_path": str(runs_root() / run_name / f"{run_name}_epoch_{epoch}_object.ckpt"),
        "goal_offset": row["goal_offset"],
        "goal_offset_steps": row["goal_offset"][-2:],
        "config_note": row.get("config_note", ""),
        "planning_mode": normalize_planning_mode(row.get("config_note", "")),
        "eval_budget": row["eval_budget"],
        "high_horizon": row["high_horizon"],
        "low_horizon": row["low_horizon"],
        "high_receding_horizon": row["high_receding_horizon"],
        "low_receding_horizon": row["low_receding_horizon"],
        "high_replan_interval": row["high_replan_interval"],
        "high_action_block": row["high_action_block"],
        "low_action_block": row["low_action_block"],
        "high_num_samples": row["high_num_samples"],
        "high_n_steps": row["high_n_steps"],
        "high_topk": row["high_topk"],
        "low_num_samples": row["low_num_samples"],
        "low_n_steps": row["low_n_steps"],
        "low_topk": row["low_topk"],
        "seed": str(seed),
        "result_file": str(result_path),
        "episode_manifest": str(manifest_path),
    }
    if isinstance(empirical, dict):
        for key, value in empirical.items():
            row_meta[f"empirical_macro_{key}"] = str(value)
    return argv, result_path, manifest_path, row_meta


def ensure_checkpoint_exists(spec: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    backend = str(spec["backend"])
    if backend == "baseline":
        path = stablewm_home() / str(checkpoint["object_relpath"])
    elif backend == "hierarchical":
        run_name = str(checkpoint["run_name"])
        epoch = str(checkpoint["checkpoint_epoch"])
        path = runs_root() / run_name / f"{run_name}_epoch_{epoch}_object.ckpt"
    else:
        raise ValueError(f"Unsupported matrix backend '{backend}'")
    if not path.exists():
        raise FileNotFoundError(
            f"Required checkpoint is missing: {path}\n"
            "Run the documented setup/checkpoint step before executing this spec."
        )


def detect_status(success_rate: str, stdout_text: str, stderr_text: str) -> str:
    if success_rate:
        return "completed"
    upper_err = stderr_text.upper()
    if "TIME LIMIT" in upper_err:
        return "time_limit"
    if "CANCELLED" in upper_err:
        return "cancelled"
    if "TRACEBACK" in stdout_text or "TRACEBACK" in stderr_text or "ERROR:" in stdout_text or "ERROR:" in stderr_text:
        return "error"
    return "missing_result"


def summarize_rows(rows: list[dict[str, str]], group_by: list[str], value_field: str) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(field, "") for field in group_by), []).append(row)

    out: list[dict[str, str]] = []
    for key, group_rows in grouped.items():
        values = [float(row[value_field]) for row in group_rows if row.get(value_field)]
        summary = {field: value for field, value in zip(group_by, key)}
        summary["num_rows"] = str(len(group_rows))
        summary["num_seeds_completed"] = str(sum(1 for row in group_rows if row.get("status") == "completed"))
        summary[f"{value_field}_mean"] = f"{statistics.mean(values):.6g}" if values else ""
        summary[f"{value_field}_std"] = f"{statistics.stdev(values):.6g}" if len(values) > 1 else ("0" if values else "")
        summary[f"{value_field}_min"] = f"{min(values):.6g}" if values else ""
        summary[f"{value_field}_max"] = f"{max(values):.6g}" if values else ""
        summary["statuses"] = ",".join(sorted({row.get("status", "") for row in group_rows if row.get("status")}))
        out.append(summary)
    return out


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ensure_parent(path)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_matrix_spec(spec: dict[str, Any], *, force: bool, dry_run: bool) -> None:
    context = context_for_spec(spec)
    done = completion_path(spec, context)
    if done is not None and done.exists() and not force:
        print(f"[skip] completion file exists: {done}")
        return

    sweep_csv = REPO_ROOT / str(spec["sweep_csv"])
    rows = read_matrix_rows(sweep_csv)
    seeds = [int(x) for x in spec.get("seeds", [42])]
    checkpoints = spec.get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("Matrix spec must define a non-empty checkpoints list")

    raw_rows: list[dict[str, str]] = []
    env = command_env(spec, context)

    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise ValueError("Each checkpoint entry must be a mapping")
        if not dry_run:
            ensure_checkpoint_exists(spec, checkpoint)
        for seed in seeds:
            for row_index, row in enumerate(rows, start=1):
                if spec["backend"] == "baseline":
                    argv, result_path, stderr_manifest, meta = build_baseline_matrix_command(
                        spec=spec,
                        checkpoint=checkpoint,
                        row=row,
                        row_index=row_index,
                        seed=seed,
                        context=context,
                    )
                else:
                    argv, result_path, stderr_manifest, meta = build_hierarchical_matrix_command(
                        spec=spec,
                        checkpoint=checkpoint,
                        row=row,
                        row_index=row_index,
                        seed=seed,
                        context=context,
                    )
                stdout_path = result_path.parent / "command.out"
                stderr_path = result_path.parent / "command.err"
                rc = run_subprocess(
                    argv,
                    env=env,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    dry_run=dry_run,
                )
                stdout_text = read_text_if_exists(stdout_path)
                stderr_text = read_text_if_exists(stderr_path)
                result_text = read_text_if_exists(result_path)
                success_rate = parse_success_rate(result_text) or parse_success_rate(stdout_text)
                status = "dry_run" if dry_run else detect_status(success_rate, stdout_text, stderr_text)
                task_row = {
                    "row_index": str(row_index),
                    "seed": str(seed),
                    "status": status,
                    "success_rate": success_rate,
                    "command_exit_code": str(rc),
                    "stdout_file": str(stdout_path),
                    "stderr_file": str(stderr_path),
                    "result_file": str(result_path),
                    "episode_manifest": meta["episode_manifest"],
                    **meta,
                    **row,
                }
                raw_rows.append(task_row)
                if rc != 0 and not dry_run:
                    raise RuntimeError(f"Matrix command failed for spec '{spec['name']}' at row {row_index}, seed {seed}")

    if dry_run:
        return

    raw_output = resolve_output_path(str(spec["outputs"]["raw_csv"]), context=context)
    raw_fields = sorted({key for row in raw_rows for key in row.keys()})
    write_csv(raw_output, raw_fields, raw_rows)

    summary_cfg = spec.get("summary", {})
    if not isinstance(summary_cfg, dict):
        raise ValueError("Matrix spec summary block must be a mapping")
    group_by = [str(item) for item in summary_cfg.get("group_by", [])]
    value_field = str(summary_cfg.get("value_field", "success_rate"))
    summary_rows = summarize_rows(raw_rows, group_by=group_by, value_field=value_field)
    summary_output = resolve_output_path(str(spec["outputs"]["summary_csv"]), context=context)
    summary_fields = list(group_by) + [
        "num_rows",
        "num_seeds_completed",
        f"{value_field}_mean",
        f"{value_field}_std",
        f"{value_field}_min",
        f"{value_field}_max",
        "statuses",
    ]
    write_csv(summary_output, summary_fields, summary_rows)
    write_completion_file(spec, context)


def run_workflow_spec(spec: dict[str, Any], *, force: bool, dry_run: bool) -> None:
    context = context_for_spec(spec)
    done = completion_path(spec, context)
    if done is not None and done.exists() and not force:
        print(f"[skip] completion file exists: {done}")
        return

    steps = spec.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError("workflow spec must define a non-empty 'steps' list")
    for step in steps:
        if not isinstance(step, str):
            raise ValueError("workflow steps must be spec-name strings")
        child_path = resolve_spec_path(step)
        child = load_yaml(child_path)
        run_spec(child, force=force, dry_run=dry_run)
    if not dry_run:
        write_completion_file(spec, context)


def run_spec(spec: dict[str, Any], *, force: bool, dry_run: bool) -> None:
    kind = str(spec.get("kind", "")).strip()
    if kind == "command":
        run_command_spec(spec, force=force, dry_run=dry_run)
        return
    if kind == "workflow":
        run_workflow_spec(spec, force=force, dry_run=dry_run)
        return
    if kind == "matrix":
        run_matrix_spec(spec, force=force, dry_run=dry_run)
        return
    raise ValueError(f"Unsupported spec kind '{kind}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a first-class H-LeWM experiment spec.")
    parser.add_argument("--spec", required=True, help="Spec name from index.yaml or a direct YAML path.")
    parser.add_argument("--force", action="store_true", help="Rerun even if the spec completion file already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec_path = resolve_spec_path(args.spec)
    spec = load_yaml(spec_path)
    run_spec(spec, force=bool(args.force), dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
