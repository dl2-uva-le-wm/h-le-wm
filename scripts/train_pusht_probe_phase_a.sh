#!/usr/bin/env bash
set -euo pipefail

exec python -m h_le_wm.experiments.run --spec probe/pusht/phase_a/train "$@"
