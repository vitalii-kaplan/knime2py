#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
WORKFLOW_REL="tests/data/KNIME_CP_10FCV_GBT"
OUT_REL="output"
# --------------------------------

IMAGE="${IMAGE:-knime2py:dev}" 

# Resolve to absolute paths relative to this script
ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKFLOW="$ROOT/$WORKFLOW_REL"
OUT="$ROOT/$OUT_REL"

# Clean and recreate output dir
rm -rf "$OUT"
mkdir -p "$OUT"

# Mirror host paths inside the container so generated code uses real paths
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$ROOT":"$ROOT" \
  -w "$ROOT" \
  "$IMAGE" \
  "$WORKFLOW" \
  --out "$OUT" \
  --graph off