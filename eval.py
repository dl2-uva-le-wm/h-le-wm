import sys

from baseline_adapter import run_baseline_script


def main() -> int:
    # Delegate baseline evaluation to the pinned upstream submodule.
    return run_baseline_script("eval.py", sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
