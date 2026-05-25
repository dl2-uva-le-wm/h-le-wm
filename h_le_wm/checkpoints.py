from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from h_le_wm.paths import REPO_ROOT, stablewm_home


REGISTRY_PATH = REPO_ROOT / "h_le_wm" / "checkpoint_registry.yaml"
KNOWN_TIERS = ("required-now", "supported-first-class", "optional-later")


def _load_registry_document(*, path: Path = REGISTRY_PATH) -> dict:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint registry must be a mapping: {path}")
    return data


def load_checkpoint_registry(*, path: Path = REGISTRY_PATH) -> list[dict[str, object]]:
    data = _load_registry_document(path=path)
    raw_entries = data.get("checkpoints", [])
    if not isinstance(raw_entries, list):
        raise ValueError("Checkpoint registry must define a list under 'checkpoints'")

    seen_names: set[str] = set()
    entries: list[dict[str, object]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            raise ValueError("Each checkpoint registry entry must be a mapping")

        name = str(item.get("name", "")).strip()
        relpath = str(item.get("relpath", "")).strip()
        raw_tiers = item.get("tiers", [])
        description = str(item.get("description", "")).strip()

        if not name:
            raise ValueError("Checkpoint registry entry is missing 'name'")
        if name in seen_names:
            raise ValueError(f"Duplicate checkpoint registry name: {name}")
        seen_names.add(name)

        if not relpath:
            raise ValueError(f"Checkpoint registry entry '{name}' is missing 'relpath'")
        relpath_obj = Path(relpath)
        if relpath_obj.is_absolute() or ".." in relpath_obj.parts:
            raise ValueError(
                f"Checkpoint registry relpath for '{name}' must be relative and stay under STABLEWM_HOME"
            )

        if not isinstance(raw_tiers, list) or not raw_tiers:
            raise ValueError(f"Checkpoint registry entry '{name}' must define a non-empty 'tiers' list")
        tiers = [str(tier).strip() for tier in raw_tiers if str(tier).strip()]
        if not tiers:
            raise ValueError(f"Checkpoint registry entry '{name}' must define at least one tier")
        unknown_tiers = sorted(set(tiers) - set(KNOWN_TIERS))
        if unknown_tiers:
            raise ValueError(
                f"Checkpoint registry entry '{name}' uses unknown tiers: {', '.join(unknown_tiers)}"
            )

        entries.append(
            {
                "name": name,
                "relpath": relpath,
                "tiers": tiers,
                "description": description,
            }
        )
    return entries


def registry_names(*, path: Path = REGISTRY_PATH) -> list[str]:
    return [str(entry["name"]) for entry in load_checkpoint_registry(path=path)]


def resolve_registry_entry(name: str, *, path: Path = REGISTRY_PATH) -> dict[str, object]:
    wanted = str(name).strip()
    for entry in load_checkpoint_registry(path=path):
        if entry["name"] == wanted:
            return entry
    raise KeyError(f"Unknown checkpoint name '{name}'. See {path}.")


def resolve_checkpoint_target(entry: dict[str, object], *, home: Path | None = None) -> Path:
    base = stablewm_home() if home is None else home.expanduser().resolve()
    return base / str(entry["relpath"])


def iter_registry_entries(
    *,
    tier: str | None = None,
    names: Sequence[str] | None = None,
    path: Path = REGISTRY_PATH,
) -> list[dict[str, object]]:
    if tier is not None and tier not in KNOWN_TIERS:
        raise ValueError(f"Unsupported checkpoint tier '{tier}'")

    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    entries = load_checkpoint_registry(path=path)

    for entry in entries:
        entry_name = str(entry["name"])
        if tier is not None and tier in entry["tiers"] and entry_name not in seen:
            selected.append(entry)
            seen.add(entry_name)

    for raw_name in names or []:
        entry = resolve_registry_entry(raw_name, path=path)
        entry_name = str(entry["name"])
        if entry_name not in seen:
            selected.append(entry)
            seen.add(entry_name)

    return selected


def parse_checkpoint_assignment(raw: str) -> tuple[str, Path]:
    name, sep, source = raw.partition("=")
    if not sep:
        raise ValueError(
            f"Invalid checkpoint assignment '{raw}'. Use the form --checkpoint name=/absolute/path/to/file"
        )
    parsed_name = name.strip()
    parsed_source = source.strip()
    if not parsed_name or not parsed_source:
        raise ValueError(
            f"Invalid checkpoint assignment '{raw}'. Use the form --checkpoint name=/absolute/path/to/file"
        )
    if parsed_source.startswith("/absolute/path/to/"):
        raise FileNotFoundError(
            "Checkpoint source still uses the docs placeholder path. Replace "
            f"'{parsed_source}' with the real local file path."
        )
    return parsed_name, Path(parsed_source).expanduser().resolve()


def stage_checkpoints(
    assignments: Sequence[str],
    *,
    force: bool,
    dry_run: bool,
    home: Path | None = None,
    path: Path = REGISTRY_PATH,
) -> list[Path]:
    base = stablewm_home() if home is None else home.expanduser().resolve()
    staged: list[Path] = []

    for raw in assignments:
        name, source = parse_checkpoint_assignment(raw)
        entry = resolve_registry_entry(name, path=path)
        target = resolve_checkpoint_target(entry, home=base)
        if not source.exists():
            raise FileNotFoundError(f"Checkpoint source does not exist: {source}")
        if target.exists() and not force:
            raise FileExistsError(
                f"Checkpoint target already exists: {target}\n"
                "Use --force to replace it."
            )

        print(f"[stage] {name}: {source} -> {target}")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        staged.append(target)
    return staged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or stage canonical H-LeWM checkpoints.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List named checkpoint slots from the machine-readable registry.")
    list_cmd.add_argument("--tier", choices=KNOWN_TIERS, default=None)

    stage = sub.add_parser("stage", help="Stage local checkpoint files into canonical STABLEWM_HOME locations.")
    stage.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint assignment in the form name=/absolute/path/to/file. Repeat as needed.",
    )
    stage.add_argument("--home", default="")
    stage.add_argument("--force", action="store_true")
    stage.add_argument("--dry-run", action="store_true")
    return parser


def _list_entries(entries: Iterable[dict[str, object]], *, home: Path | None = None) -> None:
    base = stablewm_home() if home is None else home.expanduser().resolve()
    for entry in entries:
        target = resolve_checkpoint_target(entry, home=base)
        tiers = ",".join(str(tier) for tier in entry["tiers"])
        print(f"{entry['name']}\t{tiers}\t{target}")


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list":
        entries = iter_registry_entries(tier=args.tier or None)
        _list_entries(entries)
        return 0
    if args.command == "stage":
        if not args.checkpoint:
            raise ValueError("Provide at least one --checkpoint name=/path assignment.")
        home = Path(args.home).expanduser().resolve() if args.home else None
        stage_checkpoints(
            args.checkpoint,
            force=bool(args.force),
            dry_run=bool(args.dry_run),
            home=home,
        )
        return 0
    raise RuntimeError(f"Unsupported checkpoints command '{args.command}'")


if __name__ == "__main__":
    raise SystemExit(main())
