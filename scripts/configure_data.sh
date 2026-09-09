#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/LJSpeech-1.1" >&2
  exit 2
fi

if [[ ! -d "$1" ]]; then
  echo "Dataset directory does not exist: $1" >&2
  exit 2
fi

DATASET_ROOT="$(cd "$1" && pwd)"
PREPROCESSED="$DATASET_ROOT/preprocessed_data/LJSpeech"

required=(
  "$PREPROCESSED/train.txt"
  "$PREPROCESSED/val.txt"
  "$PREPROCESSED/speakers.json"
  "$PREPROCESSED/stats.json"
  "$PREPROCESSED/mel"
  "$PREPROCESSED/pitch"
  "$PREPROCESSED/energy"
  "$PREPROCESSED/duration"
)

for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Required dataset path is missing: $path" >&2
    exit 2
  fi
done

replace_link() {
  local target="$1"
  local link="$2"
  if [[ -e "$link" && ! -L "$link" ]]; then
    echo "Refusing to replace non-symlink path: $link" >&2
    exit 2
  fi
  ln -sfn "$target" "$link"
}

mkdir -p "$PROJECT_ROOT/data"
replace_link "$DATASET_ROOT" "$PROJECT_ROOT/data/LJSpeech-1.1"
replace_link "$DATASET_ROOT/preprocessed_data" "$PROJECT_ROOT/preprocessed_data"
replace_link "$PROJECT_ROOT/preprocessed_data" "$PROJECT_ROOT/graintts/preprocessed_data"

echo "GrainTTS dataset mapping configured:"
echo "  repository: $PROJECT_ROOT"
echo "  dataset:    $DATASET_ROOT"
echo "Run: python scripts/check_setup.py"
