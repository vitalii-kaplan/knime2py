#!/usr/bin/env bash
set -euo pipefail

# run from repo root
cd "$(dirname "$0")"

PEX_BIN="${PEX_BIN:-./k2p-macos-x86_64.pex}"
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

# Active run:
"${RUNNER[@]}" tests/data/KNIME_CP_10FCV_GBT --out "$OUT" --graph off
# "${RUNNER[@]}" ../../harbour/KNIME/Payment_prediction_2022 --out "$OUT" --graph off
# "${RUNNER[@]}" ../../harbour/KNIME/HW-Churn --out "$OUT" --workbook py --graph off
# "${RUNNER[@]}" ../../harbour/KNIME/ISU_Master --out "$OUT"
# "${RUNNER[@]}" tests/data/KNIME_traverse_order --out "$OUT"
# "${RUNNER[@]}" tests/data/KNIME_PP_2022_LR --out "tests/data/!output" --graph off
# "${RUNNER[@]}" ../../harbour/KNIME/Churn_prediction_GBT --out "$OUT" --graph off
