#!/usr/bin/env bash
set -euo pipefail

# --- Config (edit these two paths relative to this script) ---
WORKFLOW_REL="tests/data/KNIME_CP_10FCV_GBT"
OUT_REL="output"
# -------------------------------------------------------------

IMAGE="${IMAGE:-ghcr.io/vitaly-chibrikov/knime2py:latest}"
WORKBOOK="${WORKBOOK:-both}"         # override: WORKBOOK=py ./run_knime_ghcr.sh

# Resolve to absolute host paths
ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKFLOW_HOST="$ROOT/$WORKFLOW_REL"
OUT_HOST="$ROOT/$OUT_REL"

# Clean and recreate output dir
rm -rf "$OUT_HOST"
mkdir -p "$OUT_HOST"

# Mirror host paths inside the container; embed real host paths in outputs
docker run --rm \
-u "$(id -u):$(id -g)" \
-v "$ROOT":"$ROOT" \
-w "$ROOT" \
"$IMAGE" \
"$WORKFLOW_HOST" \
--out "$OUT_HOST" \
--workbook "$WORKBOOK"

