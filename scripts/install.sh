#!/usr/bin/env bash
set -euo pipefail

# Installer of dependencies from pyproject.toml (run from anywhere)

# ---- Locate repo root (dir containing pyproject.toml) ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
  ROOT="$SCRIPT_DIR"
  while [[ "$ROOT" != "/" && ! -f "$ROOT/pyproject.toml" ]]; do
    ROOT="$(dirname "$ROOT")"
  done
fi
if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  echo "ERROR: Could not find pyproject.toml starting from $SCRIPT_DIR" >&2
  exit 1
fi
cd "$ROOT"

# ---- Config toggles (env vars) ----
CREATE_VENV="${CREATE_VENV:-0}"      # set to 1 to create/use .venv
VENV_DIR="${VENV_DIR:-.venv}"
WITH_DEV="${WITH_DEV:-1}"
WITH_RAG="${WITH_RAG:-1}"
WITH_ML_EXAMPLES="${WITH_ML_EXAMPLES:-1}"

# ---- Pick Python ----
PY="${PY:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY="python"

# ---- Optionally create/activate venv ----
if [[ "$CREATE_VENV" == "1" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PY" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PY="python"
fi

# ---- Install ----
$PY -m pip install -U pip
$PY -m pip install -e .

[[ "$WITH_DEV" == "1" ]] && $PY -m pip install -e ".[dev]"
[[ "$WITH_RAG" == "1" ]] && $PY -m pip install -e ".[rag]"
[[ "$WITH_ML_EXAMPLES" == "1" ]] && $PY -m pip install -e ".[ml-examples]"
