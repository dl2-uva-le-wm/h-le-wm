import sys
from pathlib import Path

from baseline_adapter import REPO_ROOT, run_baseline_script


def _has_option(args: list[str], opt: str) -> bool:
    return any(a == opt or a.startswith(f"{opt}=") for a in args)


def main() -> int:
    args = list(sys.argv[1:])

    if not _has_option(args, "--config-path"):
        args.insert(0, f"--config-path={Path(REPO_ROOT, 'config', 'eval')}")
    if not _has_option(args, "--config-name"):
        args.insert(1, "--config-name=hi_pusht")

    return run_baseline_script("eval.py", args)


if __name__ == "__main__":
    raise SystemExit(main())
