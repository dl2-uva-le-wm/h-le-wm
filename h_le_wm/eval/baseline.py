from __future__ import annotations

import sys

from h_le_wm.baseline.adapter import run_baseline_script


def main() -> int:
    return run_baseline_script("eval.py", sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
