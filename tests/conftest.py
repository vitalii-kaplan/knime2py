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
# Back-compat named workflow fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def wf_single_csv_path(workflow: Callable[[str], Path]) -> Path:
    return workflow("KNIME_single_csv")

@pytest.fixture(scope="session")
def wf_io_csv_path(workflow: Callable[[str], Path]) -> Path:
    return workflow("KNIME_io_csv")

@pytest.fixture(scope="session")
def wf_two_graphs_path(workflow: Callable[[str], Path]) -> Path:
    return workflow("KNIME_two_graphs")

@pytest.fixture(scope="session")
def wf_traverse_path(workflow: Callable[[str], Path]) -> Path:
    return workflow("KNIME_traverse_order")

# Functional flows with local paths
@pytest.fixture(scope="session")
def wf_knime_pp_2022_lr(workflow: Callable[[str], Path]) -> Path:
    return workflow("KNIME_PP_2022_LR")

@pytest.fixture(scope="session")
def wf_knime_pp_2022_dt(workflow: Callable[[str], Path]) -> Path:
    return workflow("KNIME_PP_2022_DT")

@pytest.fixture(scope="session")
def wf_knime_pp_2022_Ensemble(workflow: Callable[[str], Path]) -> Path:
    return workflow("KNIME_PP_2022_Ensemble")

# --------------------------------------------------------------------------------------
# Back-compat named node fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def node_csv_reader_dir(node_dir: Callable[[str], Path]) -> Path:
    return node_dir("Node_csv_reader")

@pytest.fixture(scope="session")
def csv_reader_node_dir(node_csv_reader_dir: Path) -> Path:
    # Alias kept for older tests
    return node_csv_reader_dir

@pytest.fixture(scope="session")
def node_csv_writer_dir(node_dir: Callable[[str], Path]) -> Path:
    return node_dir("Node_csv_writer")

@pytest.fixture(scope="session")
def node_column_filter_dir(node_dir: Callable[[str], Path]) -> Path:
    return node_dir("Node_column_filter")

@pytest.fixture(scope="session")
def node_missing_value_dir(node_dir: Callable[[str], Path]) -> Path:
    return node_dir("Node_missing_value")

@pytest.fixture(scope="session")
def node_normalizer_dir(node_dir: Callable[[str], Path]) -> Path:
    return node_dir("Node_normalizer")


# Thin wrappers to preserve old fixture names (can be removed once tests migrate)
@pytest.fixture()
def generate_lr_workbook(generate_workbook, wf_knime_pp_2022_lr: Path) -> Callable[[Path], Path]:
    def _gen(_wf: Path) -> Path:
        return generate_workbook(_wf, expected="KNIME_PP_2022_LR__g01_workbook.py")
    return _gen

@pytest.fixture()
def generate_dt_workbook(generate_workbook, wf_knime_pp_2022_dt: Path) -> Callable[[Path], Path]:
    def _gen(_wf: Path) -> Path:
        return generate_workbook(_wf, expected="KNIME_PP_2022_DT__g01_workbook.py")
    return _gen

@pytest.fixture()
def generate_ensemble_workbook(generate_workbook, wf_knime_pp_2022_Ensemble: Path) -> Callable[[Path], Path]:
    def _gen(_wf: Path) -> Path:
        return generate_workbook(_wf, expected="KNIME_PP_2022_Ensemble__g01_workbook.py")
    return _gen

@pytest.fixture()
def generate_10fcv_workbook(generate_workbook) -> Callable[[Path], Path]:
    def _gen(_wf: Path) -> Path:
        return generate_workbook(_wf, expected="KNIME_CP_10FCV_GBT__g01_workbook.py")
    return _gen
