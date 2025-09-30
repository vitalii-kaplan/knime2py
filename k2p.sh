#!/usr/bin/env bash
set -euo pipefail

# run from repo root
cd "$(dirname "$0")"

OUT="output"
mkdir -p "$OUT"
# wipe only contents
find "$OUT" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

# choose runner: prefer installed console script; fallback to module
if command -v k2p >/dev/null 2>&1; then
  RUNNER=(k2p)
else
  RUNNER=(python -m knime2py)
fi

# Active run:
#"${RUNNER[@]}" tests/data/KNIME_CP_10FCV_GBT --out "$OUT" --graph off
# "${RUNNER[@]}" ../../harbour/KNIME/Payment_prediction_2022 --out "$OUT" --graph off
# "${RUNNER[@]}" ../../harbour/KNIME/HW-Churn --out "$OUT" --workbook py --graph off
#"${RUNNER[@]}" ../../harbour/KNIME/ISU_Master_test --out "$OUT"
 "${RUNNER[@]}" tests/data/HW_Churn_test_preparation --out "$OUT"
# "${RUNNER[@]}" tests/data/KNIME_traverse_order --out "$OUT"
# "${RUNNER[@]}" tests/data/KNIME_PP_2022_LR --out "tests/data/!output" --graph off
# "${RUNNER[@]}" ../../harbour/KNIME/Churn_prediction_GBT --out "$OUT" --graph off


