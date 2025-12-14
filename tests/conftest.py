# tests/conftest.py
"""Test configuration for the knime2py project.

Overview
----------------------------
This module provides fixtures and helper functions for testing the knime2py project, 
including paths to workflows and data directories.

Runtime Behavior
----------------------------
Inputs:
- The module reads paths to workflow files and node directories based on the project 
  structure.

Outputs:
- The module provides paths to the repository root, data directory, and specific node 
  directories, which can be accessed in tests.

Edge Cases
----------------------------
The module ensures that required paths exist and raises errors if they do not. It also 
cleans up output directories before tests to avoid conflicts.

Generated Code Dependencies
----------------------------
The generated code may depend on external libraries such as pytest, but these are not 
dependencies of this module.

Usage
----------------------------
This module is typically invoked by test files to access common fixtures. For example, 
to get the path to a workflow, one might use:
```python
workflow_path = workflow('KNIME_single_csv')
```

Node Identity
----------------------------
This module does not generate code based on `settings.xml`, so there are no KNIME 
factory IDs or special flags.

Configuration
----------------------------
This module does not generate code based on `settings.xml`, so there are no dataclasses 
or configuration fields to describe.

Limitations
----------------------------
This module does not implement any specific node functionality; it serves as a 
configuration and utility module for tests.

References
----------------------------
For more information on KNIME terminology, refer to the official KNIME documentation.
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import Callable, Iterator

import pytest

# --------------------------------------------------------------------------------------
# Early coverage hook for subprocesses (pytest-cov)
# --------------------------------------------------------------------------------------
# tests/conftest.py (only the sessionstart block shown)

def pytest_sessionstart(session):
    """
    Ensure subprocess coverage is enabled for the whole test session.

    - Set COVERAGE_PROCESS_START to pyproject.toml (coverage config).
    - Prepend tests/support to PYTHONPATH so our sitecustomize shim is importable.
    - Also try pytest-cov's embed.init() (harmless if unavailable).
    """
    repo_root = Path(__file__).resolve().parents[1]
    tests_dir = Path(__file__).resolve().parent
    support_dir = tests_dir / "support"

    os.environ.setdefault("COVERAGE_PROCESS_START", str(repo_root / "pyproject.toml"))

    # Make sure our sitecustomize.py is importable by all child interpreters
    old_pp = os.environ.get("PYTHONPATH", "")
    prefix = str(support_dir)
    if old_pp:
        if not old_pp.split(os.pathsep)[0] == prefix:
            os.environ["PYTHONPATH"] = prefix + os.pathsep + old_pp
    else:
        os.environ["PYTHONPATH"] = prefix

    # Optional: keep pytest-cov embed hook too (safe if present/absent)
    try:
        from pytest_cov.embed import init  # type: ignore
        init()
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Repo paths / import path
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# Prefer: set "pythonpath = src" in pytest.ini, but keep a safe fallback here.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATA_DIR = Path(__file__).resolve().parent / "data"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _require(p: Path, msg: str) -> Path:
    """Assert a path exists with a concise error message."""
    if not p.exists():
        pytest.fail(f"{msg}: {p}")
    return p


def _workflow_path(project_dirname: str) -> Path:
    """Path to the workflow.knime for a given project directory name."""
    return DATA_DIR / project_dirname / "workflow.knime"


# --------------------------------------------------------------------------------------
# Session-scoped fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def python_exe() -> str:
    return sys.executable


# --------------------------------------------------------------------------------------
# Per-test output directory (artifact-friendly)
# --------------------------------------------------------------------------------------
@pytest.fixture()
def output_dir(data_dir: Path) -> Iterator[Path]:
    """
    Provide an empty output directory at tests/data/!output.
    Directory is cleaned before each test and left on disk after (useful for artifacts).
    """
    out_dir = data_dir / "!output"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    yield out_dir
    # Keep artifacts; uncomment to clean after each test:
    # shutil.rmtree(out_dir, ignore_errors=True)


# --------------------------------------------------------------------------------------
# Generic lookup fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def workflow() -> Callable[[str], Path]:
    """Resolve a workflow by KNIME project directory name (e.g., 'KNIME_single_csv')."""
    def _wf(name: str) -> Path:
        return _require(_workflow_path(name), "Missing sample workflow")
    return _wf


@pytest.fixture(scope="session")
def node_dir(data_dir: Path) -> Callable[[str], Path]:
    """Resolve a node directory by short name (e.g., 'Node_csv_reader')."""
    def _nd(name: str) -> Path:
        ndir = data_dir / name
        _require(ndir / "settings.xml", f"Missing node settings for {name}")
        return ndir
    return _nd


# --------------------------------------------------------------------------------------
# Common node fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def node_csv_reader_dir(node_dir: Callable[[str], Path]) -> Path:
    return node_dir("Node_csv_reader")
