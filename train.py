import sys

from baseline_adapter import run_baseline_script


def main() -> int:
    # Delegate baseline training to the pinned upstream submodule.
    return run_baseline_script("train.py", sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
