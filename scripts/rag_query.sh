#!/usr/bin/env bash
set -euo pipefail

#python -m rag.query_openai "What is this project about?"
python -m rag.query_ollama "What is this project about?"