#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./bulk_docstrings.sh [DIRECTORY]
# If DIRECTORY is not provided, defaults to "src/".

TARGET_DIR="${1:-src/}"

# Recursively find all .py files under TARGET_DIR whose basenames do NOT start with "__"
# and rewrite each file using the RAG editor.
find "$TARGET_DIR" -type f -name '*.py' ! -name '__*.py' -print0 \
| while IFS= read -r -d '' PYFILE; do
  echo "[RAG] Editing: $PYFILE"
  python -m rag.query_openai_file \
    "$PYFILE" \
    --edit "Add meaningful docstrings to all public functions based on the code." \
    --rewrite
done