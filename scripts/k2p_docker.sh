#!/usr/bin/env bash
set -euo pipefail

# scripts/k2p_docker.sh — run knime2py from GHCR inside Docker
# Usage: ./scripts/k2p_docker.sh

# --- Config (edit these two paths relative to the repo root) ---
WORKFLOW_REL="${WORKFLOW_REL:-tests/data/KNIME_CP_10FCV_GBT}"
OUT_REL="${OUT_REL:-output}"
# ---------------------------------------------------------------

IMAGE="${IMAGE:-ghcr.io/vitalii-kaplan/knime2py:latest}"

# Locate repo root (works whether this script is called from anywhere)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

WORKFLOW_HOST="$ROOT/$WORKFLOW_REL"
OUT_HOST="$ROOT/$OUT_REL"

# Clean and recreate output dir
rm -rf "$OUT_HOST"
mkdir -p "$OUT_HOST"

# Mirror host paths inside the container so generated code embeds real host paths
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$ROOT":"$ROOT" \
  -w "$ROOT" \
  "$IMAGE" \
  "$WORKFLOW_HOST" \
  --out "$OUT_HOST"
