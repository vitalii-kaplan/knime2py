# conftest.py
from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterator, Optional

import pytest

# --------------------------------------------------------------------------------------
# Repo paths
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
# Prefer: put `pythonpath = src` into pytest.ini and delete this block.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path(__file__).resolve().parent / "data"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _require(p: Path, msg: str) -> Path:
    """Assert a path exists with a concise error."""
    if not p.exists():
        pytest.fail(f"{msg}: {p}")
    return p


def _workflow_path(project_dirname: str) -> Path:
    """tests/data/<project_dirname>/workflow.knime"""
    return DATA_DIR / project_dirname / "workflow.knime"


# --------------------------------------------------------------------------------------
# Common helpers / fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def python_exe() -> str:
    """Path to the current Python interpreter."""
    return sys.executable

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
    # Keep artifacts; uncomment to clean post-test:
    # shutil.rmtree(out_dir, ignore_errors=True)

# --------------------------------------------------------------------------------------
# Generic lookup fixtures (reduce duplication)
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def workflow() -> Callable[[str], Path]:
    """
    Resolve a workflow by KNIME project directory name.
    Example: workflow('KNIME_single_csv')
    """
    def _wf(name: str) -> Path:
        return _require(_workflow_path(name), "Missing sample workflow")
    return _wf


@pytest.fixture(scope="session")
def node_dir(data_dir: Path) -> Callable[[str], Path]:
    """
    Resolve a node directory by short name.
    Example: node_dir('Node_csv_reader') → tests/data/Node_csv_reader
    """
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

