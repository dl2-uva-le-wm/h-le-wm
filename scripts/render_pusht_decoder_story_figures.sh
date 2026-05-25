#!/usr/bin/env bash
set -euo pipefail

exec python -m h_le_wm.experiments.run --spec render/pusht/decoder_story_figures "$@"
