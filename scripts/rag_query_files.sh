#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./rag_query_files.sh [DIRECTORY]
# If DIRECTORY is not provided, defaults to TARGET_DIR.

TARGET_DIR="${1:-tests/}"

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