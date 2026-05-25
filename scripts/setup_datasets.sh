#!/usr/bin/env bash

# Usage examples:
#   source scripts/setup_datasets.sh
#   source scripts/setup_datasets.sh --home /absolute/path/to/data
#   source scripts/setup_datasets.sh --datasets pusht,cube
#
# Notes:
# - Prefer sourcing so STABLEWM_HOME is exported in your current shell.
# - If executed (not sourced), downloads still work, but export will not persist.

if [[ -n "${BASH_VERSION:-}" ]]; then
  _THIS_FILE="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _THIS_FILE="${(%):-%N}"
else
  _THIS_FILE="$0"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${_THIS_FILE}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
DEFAULT_HOME="${ROOT_DIR}/data/stablewm"

HOME_DIR="${DEFAULT_HOME}"
DATASETS="all"

resolve_python_bin() {
  local candidate
  for candidate in "${PYTHON:-}" python3 python; do
    if [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Missing required command: $name" >&2
    return 1
  fi
}

ensure_zstd_support() {
  if command -v zstd >/dev/null 2>&1; then
    return 0
  fi

  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
raise SystemExit(0 if importlib.util.find_spec("zstandard") is not None else 1)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --home)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --home" >&2
        return 1 2>/dev/null || exit 1
      fi
      HOME_DIR="$2"
      shift 2
      ;;
    --datasets)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --datasets" >&2
        return 1 2>/dev/null || exit 1
      fi
      DATASETS="$2"
      shift 2
      ;;
    -h|--help)
      cat <<USAGE
Usage: source scripts/setup_datasets.sh [--home PATH] [--datasets LIST]

Options:
  --home PATH       Set STABLEWM_HOME (default: ${DEFAULT_HOME})
  --datasets LIST   Comma-separated: pusht,tworooms,cube,reacher,all
USAGE
      return 0 2>/dev/null || exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      return 1 2>/dev/null || exit 1
      ;;
  esac
done

PYTHON_BIN="$(resolve_python_bin)" || {
  echo "Could not find a usable Python interpreter. Set \$PYTHON or install python/python3." >&2
  return 1 2>/dev/null || exit 1
}

require_command curl || return 1 2>/dev/null || exit 1
require_command tar || return 1 2>/dev/null || exit 1
ensure_zstd_support || {
  echo "Missing zstd support: install the 'zstd' CLI or the Python 'zstandard' package." >&2
  return 1 2>/dev/null || exit 1
}

# Export for this shell session (persists only when sourced).
export STABLEWM_HOME="${HOME_DIR}"
mkdir -p "${STABLEWM_HOME}"

echo "STABLEWM_HOME=${STABLEWM_HOME}"

# Resolve dataset repos from short names.
# Official collection: https://huggingface.co/collections/quentinll/lewm
REPOS=()

normalize_dataset() {
  local x
  x="$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  case "$x" in
    pusht) echo "quentinll/lewm-pusht" ;;
    tworoom|tworooms) echo "quentinll/lewm-tworooms" ;;
    cube) echo "quentinll/lewm-cube" ;;
    reacher) echo "quentinll/lewm-reacher" ;;
    all) echo "all" ;;
    *) echo "" ;;
  esac
}

if [[ "${DATASETS}" == "all" ]]; then
  REPOS=(
    "quentinll/lewm-pusht"
    "quentinll/lewm-tworooms"
    "quentinll/lewm-cube"
    "quentinll/lewm-reacher"
  )
else
  dataset_items=()
  if [[ -n "${BASH_VERSION:-}" ]]; then
    IFS=',' read -r -a dataset_items <<< "${DATASETS}"
  elif [[ -n "${ZSH_VERSION:-}" ]]; then
    IFS=',' read -rA dataset_items <<< "${DATASETS}"
  else
    OLD_IFS="${IFS}"
    IFS=','
    # shellcheck disable=SC2086
    set -- ${DATASETS}
    IFS="${OLD_IFS}"
    dataset_items=("$@")
  fi

  for item in "${dataset_items[@]}"; do
    repo="$(normalize_dataset "$item")"
    if [[ -z "$repo" || "$repo" == "all" ]]; then
      echo "Unsupported dataset key: '$item'. Use: pusht,tworooms,cube,reacher,all" >&2
      return 1 2>/dev/null || exit 1
    fi
    REPOS+=("$repo")
  done
fi

fetch_repo_files() {
  local repo="$1"
  REPO="$repo" "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
import urllib.request

repo = os.environ["REPO"]
url = f"https://huggingface.co/api/datasets/{repo}"
try:
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
except Exception as e:
    print(f"ERROR: cannot query {url}: {e}", file=sys.stderr)
    sys.exit(2)

if isinstance(data, list):
    if not data:
        sys.exit(0)
    data = data[0]
if not isinstance(data, dict):
    print(f"ERROR: unexpected API response type: {type(data)}", file=sys.stderr)
    sys.exit(2)

# Keep dataset payload files only.
# We intentionally include both compressed and uncompressed forms.
allowed_suffixes = (
    ".h5", ".hdf5", ".zst", ".tar", ".tar.gz", ".tgz"
)
for s in data.get("siblings", []):
    name = s.get("rfilename", "")
    if name.endswith(allowed_suffixes):
        print(name)
PY
}

download_file() {
  local repo="$1"
  local relpath="$2"
  local out_path="${STABLEWM_HOME}/${relpath}"

  mkdir -p "$(dirname "$out_path")"
  if [[ -f "$out_path" ]]; then
    echo "[skip] already exists: $out_path"
    return 0
  fi

  local url="https://huggingface.co/datasets/${repo}/resolve/main/${relpath}?download=true"
  echo "[download] $repo/$relpath"
  curl -L --fail --progress-bar "$url" -o "$out_path" || {
    echo "Download failed: $url" >&2
    return 1
  }
}

extract_zstd_file() {
  local source_path="$1"
  local target_path="$2"

  if command -v zstd >/dev/null 2>&1; then
    zstd -d --rm -f "$source_path" -o "$target_path" || {
      echo "zstd extract failed: $source_path" >&2
      return 1
    }
    return 0
  fi

  "$PYTHON_BIN" - "$source_path" "$target_path" <<'PY'
import os
import sys

try:
    import zstandard as zstd
except ModuleNotFoundError:
    print(
        "Missing zstd support: install the 'zstd' CLI or the Python 'zstandard' package.",
        file=sys.stderr,
    )
    raise SystemExit(1)

source_path, target_path = sys.argv[1], sys.argv[2]
with open(source_path, "rb") as src, open(target_path, "wb") as dst:
    dctx = zstd.ZstdDecompressor()
    dctx.copy_stream(src, dst)
os.remove(source_path)
PY
}

extract_if_needed() {
  local path="$1"

  case "$path" in
    *.h5.zst|*.hdf5.zst)
      local target="${path%.zst}"
      if [[ -f "$target" ]]; then
        echo "[skip] extracted exists: $target"
      else
        echo "[extract] $path -> $target"
        extract_zstd_file "$path" "$target" || return 1
      fi
      ;;
    *.tar.zst)
      local tar_path="${path%.zst}"
      if [[ ! -f "$tar_path" ]]; then
        echo "[extract] $path -> $tar_path"
        extract_zstd_file "$path" "$tar_path" || return 1
      fi
      echo "[untar] $tar_path"
      tar -xf "$tar_path" -C "$STABLEWM_HOME" || {
        echo "tar extract failed: $tar_path" >&2
        return 1
      }
      ;;
    *.tar)
      echo "[untar] $path"
      tar -xf "$path" -C "$STABLEWM_HOME" || {
        echo "tar extract failed: $path" >&2
        return 1
      }
      ;;
    *.tar.gz|*.tgz)
      echo "[untar] $path"
      tar -xzf "$path" -C "$STABLEWM_HOME" || {
        echo "tar extract failed: $path" >&2
        return 1
      }
      ;;
  esac
}

for repo in "${REPOS[@]}"; do
  echo ""
  echo "==> Scanning ${repo}"
  files_text="$(fetch_repo_files "$repo")" || {
    echo "Failed to list files for ${repo}" >&2
    return 1 2>/dev/null || exit 1
  }
  files=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && files+=("$line")
  done <<< "$files_text"

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No dataset payload files found in ${repo}" >&2
    return 1 2>/dev/null || exit 1
  fi

  for f in "${files[@]}"; do
    download_file "$repo" "$f" || return 1 2>/dev/null || exit 1
    extract_if_needed "${STABLEWM_HOME}/${f}" || return 1 2>/dev/null || exit 1
  done

done

echo ""
echo "Done. STABLEWM_HOME is set to: ${STABLEWM_HOME}"
echo "Tip: add this to ~/.zshrc to persist:"
echo "  export STABLEWM_HOME=\"${STABLEWM_HOME}\""
