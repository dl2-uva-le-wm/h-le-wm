#!/usr/bin/env bash

# Usage examples:
#   source scripts/setup_datasets.sh
#   source scripts/setup_datasets.sh --home /absolute/path/to/data
#   source scripts/setup_datasets.sh --datasets pusht,cube
#
# Notes:
# - Prefer sourcing so STABLEWM_HOME is exported in your current shell.
# - If executed (not sourced), downloads still work, but export will not persist.

# Detect whether the script is sourced (bash/zsh).
_IS_SOURCED=0
if [[ -n "${BASH_VERSION:-}" ]]; then
  [[ "${BASH_SOURCE[0]}" != "$0" ]] && _IS_SOURCED=1
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  case $ZSH_EVAL_CONTEXT in *:file) _IS_SOURCED=1 ;; esac
fi

_die() {
  echo "$1" >&2
  return 1
}

_abort() {
  _die "$1"
  return 1
}

# Works when sourced from bash or zsh.
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --home)
      if [[ $# -lt 2 ]]; then
        _abort "Missing value for --home"
        return 1 2>/dev/null || exit 1
      fi
      HOME_DIR="$2"
      shift 2
      ;;
    --datasets)
      if [[ $# -lt 2 ]]; then
        _abort "Missing value for --datasets"
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
      _abort "Unknown argument: $1"
      return 1 2>/dev/null || exit 1
      ;;
  esac
done

# Export for this shell session (persists only when sourced).
export STABLEWM_HOME="${HOME_DIR}"
mkdir -p "${STABLEWM_HOME}"

echo "STABLEWM_HOME=${STABLEWM_HOME}"

# Resolve dataset repos from short names.
# Official collection: https://huggingface.co/collections/quentinll/lewm
REPOS=()

normalize_dataset() {
  local x
  x="$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
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
  for item in ${DATASETS//,/ }; do
    repo="$(normalize_dataset "$item")"
    if [[ -z "$repo" || "$repo" == "all" ]]; then
      _abort "Unsupported dataset key: '$item'. Use: pusht,tworooms,cube,reacher,all"
      return 1 2>/dev/null || exit 1
    fi
    REPOS+=("$repo")
  done
fi

fetch_repo_files() {
  local repo="$1"
  REPO="$repo" python3 - <<'PY'
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
        size = s.get("size")
        if size is None:
            lfs = s.get("lfs")
            if isinstance(lfs, dict):
                size = lfs.get("size")
        if isinstance(size, int):
            print(f"{name}\t{size}")
        else:
            print(name)
PY
}

file_size_bytes() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return 1
  fi

  stat -c %s "$path" 2>/dev/null || stat -f %z "$path" 2>/dev/null
}

archive_extract_target() {
  local path="$1"

  case "$path" in
    *.h5.zst|*.hdf5.zst|*.tar.zst)
      echo "${path%.zst}"
      ;;
    *)
      return 1
      ;;
  esac
}

archive_needs_validation() {
  local path="$1"
  local target=""

  target="$(archive_extract_target "$path")" || return 1
  [[ ! -f "$target" ]]
}

validate_archive() {
  local path="$1"

  case "$path" in
    *.zst)
      zstd -t -q "$path" >/dev/null 2>&1
      ;;
    *.tar.gz|*.tgz)
      gzip -t "$path" >/dev/null 2>&1
      ;;
    *)
      return 0
      ;;
  esac
}

download_file() {
  local repo="$1"
  local relpath="$2"
  local expected_size="${3:-}"
  local out_path="${STABLEWM_HOME}/${relpath}"
  local part_path="${out_path}.part"
  local current_size=""
  local part_size=""

  mkdir -p "$(dirname "$out_path")"
  if [[ -f "$out_path" ]]; then
    if [[ -n "$expected_size" ]] && current_size="$(file_size_bytes "$out_path")"; then
      if [[ "$current_size" == "$expected_size" ]]; then
        if archive_needs_validation "$out_path" && ! validate_archive "$out_path"; then
          echo "[redownload] corrupt archive detected: $out_path"
          rm -f "$out_path" "$part_path"
        else
          echo "[skip] already exists: $out_path"
          return 0
        fi
      fi

      if [[ -f "$out_path" ]]; then
        echo "[resume] detected incomplete file (${current_size}/${expected_size} bytes): $out_path"
        if [[ -f "$part_path" ]]; then
          if part_size="$(file_size_bytes "$part_path")" && [[ "$part_size" -ge "$current_size" ]]; then
            rm -f "$out_path"
          else
            mv -f "$out_path" "$part_path"
          fi
        else
          mv -f "$out_path" "$part_path"
        fi
      fi
    else
      if archive_needs_validation "$out_path" && ! validate_archive "$out_path"; then
        echo "[redownload] corrupt archive detected: $out_path"
        rm -f "$out_path" "$part_path"
      else
        echo "[skip] already exists: $out_path"
        return 0
      fi
    fi
  fi

  if [[ -f "$part_path" ]] && [[ -n "$expected_size" ]] && part_size="$(file_size_bytes "$part_path")"; then
    if [[ "$part_size" == "$expected_size" ]]; then
      mv -f "$part_path" "$out_path"
      echo "[resume] recovered completed file: $out_path"
      return 0
    fi
  fi

  local url="https://huggingface.co/datasets/${repo}/resolve/main/${relpath}?download=true"
  echo "[download] $repo/$relpath"

  local -a curl_cmd=(
    curl
    -L
    --fail
    --progress-bar
    --retry 5
    --retry-delay 5
    --retry-all-errors
  )
  if [[ -f "$part_path" ]]; then
    if part_size="$(file_size_bytes "$part_path")"; then
      echo "[resume] continuing partial download (${part_size} bytes present): $part_path"
    else
      echo "[resume] continuing partial download: $part_path"
    fi
    curl_cmd+=(--continue-at -)
  fi

  "${curl_cmd[@]}" -o "$part_path" "$url" || {
    _abort "Download failed: $url"
    return 1
  }

  if [[ -n "$expected_size" ]] && current_size="$(file_size_bytes "$part_path")"; then
    if [[ "$current_size" != "$expected_size" ]]; then
      _abort "Downloaded size mismatch for ${relpath}: got ${current_size}, expected ${expected_size}"
      return 1
    fi
  fi

  mv -f "$part_path" "$out_path" || {
    _abort "Failed to finalize download: $out_path"
    return 1
  }
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
        zstd -d --rm -f "$path" -o "$target" || {
          rm -f "$target"
          _abort "zstd extract failed: $path"
          return 1
        }
      fi
      ;;
    *.tar.zst)
      local tar_path="${path%.zst}"
      if [[ ! -f "$tar_path" ]]; then
        echo "[extract] $path -> $tar_path"
        zstd -d --rm -f "$path" -o "$tar_path" || {
          rm -f "$tar_path"
          _abort "zstd extract failed: $path"
          return 1
        }
      fi
      echo "[untar] $tar_path"
      tar -xf "$tar_path" -C "$STABLEWM_HOME" || {
        _abort "tar extract failed: $tar_path"
        return 1
      }
      ;;
    *.tar)
      echo "[untar] $path"
      tar -xf "$path" -C "$STABLEWM_HOME" || {
        _abort "tar extract failed: $path"
        return 1
      }
      ;;
    *.tar.gz|*.tgz)
      echo "[untar] $path"
      tar -xzf "$path" -C "$STABLEWM_HOME" || {
        _abort "tar extract failed: $path"
        return 1
      }
      ;;
  esac
}

for repo in "${REPOS[@]}"; do
  echo ""
  echo "==> Scanning ${repo}"
  files_text="$(fetch_repo_files "$repo")" || {
    _abort "Failed to list files for ${repo}"
    return 1 2>/dev/null || exit 1
  }
  files=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && files+=("$line")
  done <<< "$files_text"

  if [[ ${#files[@]} -eq 0 ]]; then
    _abort "No dataset payload files found in ${repo}"
    return 1 2>/dev/null || exit 1
  fi

  for file_spec in "${files[@]}"; do
    IFS=$'\t' read -r f size <<< "$file_spec"
    download_file "$repo" "$f" "$size" || return 1 2>/dev/null || exit 1
    extract_if_needed "${STABLEWM_HOME}/${f}" || return 1 2>/dev/null || exit 1
  done

done

echo ""
echo "Done. STABLEWM_HOME is set to: ${STABLEWM_HOME}"
echo "Tip: add this to ~/.zshrc to persist:"
echo "  export STABLEWM_HOME=\"${STABLEWM_HOME}\""
