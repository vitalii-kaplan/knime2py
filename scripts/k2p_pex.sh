#!/usr/bin/env bash
set -euo pipefail

# scripts/k2p_pex.sh — run the locally built PEX
# Usage: ./scripts/k2p_pex.sh

# ----- Locate repo root (works from anywhere) -----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$ROOT"

# ----- Pick a default PEX by OS/arch; allow override via PEX_BIN -----
case "$(uname -s)" in
  Darwin)  PEX_DEFAULT="dist/k2p-macos-$(uname -m).pex" ;;  # arm64 or x86_64
  Linux)   PEX_DEFAULT="dist/k2p-linux.pex" ;;
  MINGW*|MSYS*|CYGWIN*) PEX_DEFAULT="dist/k2p-windows.pex" ;;
  *)       PEX_DEFAULT="dist/k2p-unknown.pex" ;;
esac
PEX_BIN="${PEX_BIN:-$PEX_DEFAULT}"

if [[ ! -f "$PEX_BIN" ]]; then
  echo "PEX not found: $PEX_BIN" >&2
  echo "Available files in dist/:"
  ls -1 dist 2>/dev/null || true
  exit 1
fi

OUT="output"
mkdir -p "$OUT"
# wipe only contents
find "$OUT" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

# Prefer python3; fallback to python
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python

# Run
"$PY" "$PEX_BIN" tests/data/KNIME_CP_10FCV_GBT --out "$OUT"
