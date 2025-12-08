#!/usr/bin/env bash
set -euo pipefail

# Install your package and its runtime dependencies
python -m pip install .

# Install dev dependencies (pytest, pytest-cov, etc.)
python -m pip install ".[dev]"

# Run tests with coverage, using your real package name
pytest --cov=./ --cov-report=xml:coverage.xml
