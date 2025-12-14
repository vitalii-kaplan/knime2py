# tests/support/sitecustomize.py
"""
Shim to start coverage in child interpreters.

Python imports `site`, which then auto-imports `sitecustomize` if present on sys.path.
If COVERAGE_PROCESS_START is set, call coverage.process_startup() so subprocesses
participate in coverage just like the main pytest process.
"""
from __future__ import annotations
import os

def _maybe_start_coverage() -> None:
    cfg = os.environ.get("COVERAGE_PROCESS_START")
    if not cfg:
        return
    try:
        import coverage  # provided by coverage.py
        coverage.process_startup()
    except Exception:
        # Don't break tests if coverage is missing in some environments.
        pass

_maybe_start_coverage()
