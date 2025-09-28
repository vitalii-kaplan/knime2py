#!/usr/bin/env bash
set -euo pipefail

# test_gen.sh — wrapper for the cleanup step (src/test_gen/cli.py)
# Run with no arguments. Configure the target KNIME project below.

#####################################
# CONFIG — pick ONE of these:
#####################################

# Option A: Clean a project by NAME located under tests/data/<NAME>
WORKFLOW_NAME="ISU_Master_test_preparation"   # e.g., "KNIME_io_csv" or "KNIME_PP_2022_LR"
DATA_DIR="tests/data"          # change if your tests data dir differs

# Option B: Clean an explicit KNIME project directory (absolute or relative)
# If non-empty, this takes precedence over WORKFLOW_NAME.
KNIME_PROJECT_PATH=""          # e.g., "../../harbour/KNIME/ISU_Master_test_preparation"

# Behavior flags
DRY_RUN=0                      # 1 = show what would be deleted; 0 = actually delete
VERBOSE=1                      # 1 = print details; 0 = quiet

#####################################
# Script (no edits needed below)
#####################################

# Move to repo root (assumes this script lives in the repo root)
cd "$(dirname "$0")"

# Ensure local 'src' is importable for `python -m test_gen.cli`
export PYTHONPATH="${PYTHONPATH:-}:src"

# Pick Python
PY_BIN="python3"
command -v python3 >/dev/null 2>&1 || PY_BIN="python"

# Build command
CMD=("$PY_BIN" -m test_gen.cli)

if [[ -n "${KNIME_PROJECT_PATH}" ]]; then
  # Explicit path mode
  CMD+=(--path "$KNIME_PROJECT_PATH")
elif [[ -n "${WORKFLOW_NAME}" ]]; then
  # Name under tests/data mode
  CMD+=("$WORKFLOW_NAME" --data-dir "$DATA_DIR")
else
  echo "ERROR: Set either KNIME_PROJECT_PATH or WORKFLOW_NAME in the CONFIG section." >&2
  exit 2
fi

[[ "$DRY_RUN" -eq 1 ]] && CMD+=(--dry-run)
[[ "$VERBOSE" -eq 1 ]] && CMD+=(-v)

echo "+ ${CMD[*]}" >&2
exec "${CMD[@]}"
