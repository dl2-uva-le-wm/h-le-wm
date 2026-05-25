#!/usr/bin/env bash

# Canonical paper-facing dataset setup wrapper.
# Usage:
#   source scripts/setup_paper_datasets.sh
#   source scripts/setup_paper_datasets.sh --home /absolute/path/to/stablewm

_paper_die() {
  echo "$1" >&2
  return 1
}

if [[ -n "${BASH_VERSION:-}" ]]; then
  _PAPER_THIS_FILE="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _PAPER_THIS_FILE="${(%):-%N}"
else
  _PAPER_THIS_FILE="$0"
fi

_PAPER_SCRIPT_DIR="$(cd -- "$(dirname -- "${_PAPER_THIS_FILE}")" >/dev/null 2>&1 && pwd)"

for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    cat <<'USAGE'
Usage: source scripts/setup_paper_datasets.sh [--home PATH]

Canonical paper dataset setup.

Options:
  --home PATH   Set STABLEWM_HOME before downloading

Datasets staged by this wrapper:
  pusht,cube
USAGE
    return 0 2>/dev/null || exit 0
  fi
  if [[ "$arg" == "--datasets" ]]; then
    _paper_die "scripts/setup_paper_datasets.sh always installs the paper datasets: pusht,cube"
    return 1 2>/dev/null || exit 1
  fi
done

source "${_PAPER_SCRIPT_DIR}/setup_datasets.sh" --datasets "pusht,cube" "$@"
