#!/usr/bin/env bash
set -euo pipefail

# run from repo root
cd "$(dirname "$0")"

PEX_BIN="${PEX_BIN:-./dist/k2p-macos-arm64.pex}"
if [[ ! -f "$PEX_BIN" ]]; then
  echo "PEX not found: $PEX_BIN" >&2
  exit 1
fi

OUT="output"
mkdir -p "$OUT"
# wipe only contents
find "$OUT" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

# Always run the PEX explicitly with python3 (works even if not executable)
RUNNER=(python3 "$PEX_BIN")

# Run:
"${RUNNER[@]}" tests/data/KNIME_CP_10FCV_GBT --out "$OUT"
