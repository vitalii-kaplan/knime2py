#!/usr/bin/env bash
set -euo pipefail

rm -rf .rag_index/

python -m rag.snapshot_structure
python -m rag.index_sbert --full
python -m rag.index_openai --full