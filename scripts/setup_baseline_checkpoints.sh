#!/usr/bin/env bash
set -euo pipefail

exec python -m h_le_wm.checkpoints stage "$@"
