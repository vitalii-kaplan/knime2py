#!/usr/bin/env bash
set -euo pipefail

git tag -a v0.1.11 -m "v0.1.11
- Add RAG index and request tooling.
- Generate module docstrings for all Python files (RAG-assisted).
- Build and publish documentation with MkDocs."

git push origin v0.1.11