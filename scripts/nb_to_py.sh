#!/usr/bin/env bash
# Convert notebooks to py:percent for cheap reading without altering originals.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-/tmp/opencode/nb_py_mirror}"
mkdir -p "$OUT"

find "$ROOT/notebooks" -name '*.ipynb' -print0 | while IFS= read -r -d '' nb; do
  rel="${nb#$ROOT/notebooks/}"
  target="$OUT/${rel%.ipynb}.py"
  mkdir -p "$(dirname "$target")"
  jupytext --to py:percent --output "$target" "$nb" >/dev/null
  echo "$target"
done
