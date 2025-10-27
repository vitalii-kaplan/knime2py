#!/usr/bin/env bash
set -euo pipefail

#python -m rag.query_openai "What is this project about?"
#python -m rag.query_ollama "What is this project about?"
python -m rag.query_openai_file "src/knime2py/emitters.py" --edit "Add meaningful docstrings to all public functions based on the code." --raw