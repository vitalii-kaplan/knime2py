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
    """Assert a path exists with a concise error message."""
    if not p.exists():
        pytest.fail(f"{msg}: {p}")
    return p


def _workflow_path(project_dirname: str) -> Path:
    """Construct the path to the workflow file for a given project directory name.

    Args:
        project_dirname (str): The name of the project directory.

    Returns:
        Path: The path to the workflow.knime file.
    """
    return DATA_DIR / project_dirname / "workflow.knime"


# --------------------------------------------------------------------------------------
# Common helpers / fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Provide the root path of the repository.

    Returns:
        Path: The root path of the repository.
    """
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Provide the path to the data directory.

    Returns:
        Path: The path to the data directory.
    """
    return DATA_DIR


@pytest.fixture(scope="session")
def python_exe() -> str:
    """Provide the path to the current Python interpreter.

    Returns:
        str: The path to the current Python executable.
    """
    return sys.executable

@pytest.fixture()
def output_dir(data_dir: Path) -> Iterator[Path]:
    """Provide an empty output directory for test artifacts.

    The directory is cleaned before each test and left on disk after the test.

    Args:
        data_dir (Path): The path to the data directory.

    Yields:
        Iterator[Path]: The path to the output directory.
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
    """Resolve a workflow by KNIME project directory name.

    Example:
        workflow('KNIME_single_csv')

    Returns:
        Callable[[str], Path]: A function that returns the path to the workflow.
    """
    def _wf(name: str) -> Path:
        return _require(_workflow_path(name), "Missing sample workflow")
    return _wf


@pytest.fixture(scope="session")
def node_dir(data_dir: Path) -> Callable[[str], Path]:
    """Resolve a node directory by short name.

    Example:
        node_dir('Node_csv_reader') → tests/data/Node_csv_reader

    Args:
        data_dir (Path): The path to the data directory.

    Returns:
        Callable[[str], Path]: A function that returns the path to the node directory.
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
    """Provide the path to the CSV reader node directory.

    Args:
        node_dir (Callable[[str], Path]): A function to resolve node directories.

    Returns:
        Path: The path to the CSV reader node directory.
    """
    return node_dir("Node_csv_reader")
