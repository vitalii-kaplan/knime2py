#!/usr/bin/env bash
set -euo pipefail

# scripts/k2p.sh — run knime2py from a local install (console script or module)
# Usage: ./scripts/k2p.sh

# ----- Locate repo root (works from anywhere) -----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$ROOT"

OUT="${OUT:-output}"
mkdir -p "$OUT"
# wipe only contents
find "$OUT" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

# choose runner: prefer installed console script; fallback to module
if command -v k2p >/dev/null 2>&1; then
  RUNNER=(k2p)
else
  # prefer python3 if available
  PY=python3; command -v python3 >/dev/null 2>&1 || PY=python
  RUNNER=("$PY" -m knime2py)
fi

# Active run (pick one; current selection is HW_Churn_test):
# "${RUNNER[@]}" tests/data/KNIME_CP_10FCV_GBT --out "$OUT" --graph off
# "${RUNNER[@]}" ../../harbour/KNIME/Payment_prediction_2022 --out "$OUT" --graph off
# "${RUNNER[@]}" ../../harbour/KNIME/HW-Churn --out "$OUT" --workbook py --graph off
# "${RUNNER[@]}" ../../harbour/KNIME/ISU_Master_test --out "$OUT"
"${RUNNER[@]}" tests/data/HW_Churn_test --out "$OUT" --graph off --workbook ipynb
# "${RUNNER[@]}" tests/data/KNIME_traverse_order --out "$OUT"
# "${RUNNER[@]}" tests/data/KNIME_PP_2022_LR --out "tests/data/!output" --graph off
# "${RUNNER[@]}" ../../harbour/KNIME/Churn_prediction_GBT --out "$OUT" --graph off
