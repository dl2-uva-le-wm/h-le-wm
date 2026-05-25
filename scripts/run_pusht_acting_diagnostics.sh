#!/usr/bin/env bash

if [[ -n "${BASH_VERSION:-}" ]]; then
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "Do not source scripts/run_pusht_acting_diagnostics.sh; run it as a command instead." >&2
    return 1 2>/dev/null || exit 1
  fi
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  case $ZSH_EVAL_CONTEXT in
    *:file)
      echo "Do not source scripts/run_pusht_acting_diagnostics.sh; run it as a command instead." >&2
      return 1 2>/dev/null || exit 1
      ;;
  esac
fi

set -euo pipefail
exec python -m h_le_wm.experiments.run --spec diagnostics/pusht/acting "$@"
