#!/usr/bin/env bash
set -euo pipefail

python -m pip install "mkdocs>=1.6" "mkdocs-material" "mkdocstrings[python]" \
                       "mkdocs-gen-files" "mkdocs-literate-nav" "mkdocs-section-index"


mkdocs serve