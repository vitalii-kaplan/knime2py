# conftest.py
from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Optional

import pytest

# --------------------------------------------------------------------
# Resolve repo root and make package importable in tests
# --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path(__file__).resolve().parent / "data"


def _workflow_path(project_dirname: str) -> Path:
    """tests/data/<project_dirname>/workflow.knime"""
    return DATA_DIR / project_dirname / "workflow.knime"


# --------------------------------------------------------------------
# Common helpers / fixtures
# --------------------------------------------------------------------
@pytest.fixture(scope="session")
def python_exe() -> str:
    """Path to the current Python interpreter."""
    return sys.executable


@pytest.fixture(scope="session")
def k2p_script() -> Path:
    """Path to k2p.py at the repo root."""
    script = REPO_ROOT / "k2p.py"
    if not script.exists():
        pytest.fail(f"Cannot find k2p.py at {script}")
    return script


@pytest.fixture()
def clean_output_dir() -> Iterator[Path]:
    """
    Provide a clean output directory at tests/data/!output (as requested).
    The directory is deleted before the test and left on disk after (useful for artifacts).
    """
    out_dir = DATA_DIR / "!output"
    # nuke any prior runs
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    yield out_dir
    # Do not delete after test to allow inspection; uncomment to clean:
    # shutil.rmtree(out_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def run_cli():
    """
    Helper to run the CLI: python k2p.py <args...>
    Returns a function that executes the command and returns CompletedProcess.
    """
    def _run(
        script: Path,
        args: list[str],
        cwd: Optional[Path] = None,
        check: bool = True,
        env: Optional[dict] = None,
    ) -> subprocess.CompletedProcess:
        cmd = [sys.executable, str(script), *args]
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env or os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if check and proc.returncode != 0:
            raise AssertionError(
                f"CLI failed ({proc.returncode}).\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )
        return proc

    return _run


# --------------------------------------------------------------------
# Workflow fixtures (existing + new)
# --------------------------------------------------------------------
@pytest.fixture(scope="session")
def wf_single_csv_path() -> Path:
    wf = _workflow_path("KNIME_single_csv")
    if not wf.exists():
        pytest.fail(f"Missing sample workflow: {wf}")
    return wf


@pytest.fixture(scope="session")
def wf_io_csv_path() -> Path:
    wf = _workflow_path("KNIME_io_csv")
    if not wf.exists():
        pytest.fail(f"Missing sample workflow: {wf}")
    return wf


@pytest.fixture(scope="session")
def wf_two_graphs_path() -> Path:
    wf = _workflow_path("KNIME_two_graphs")
    if not wf.exists():
        pytest.fail(f"Test data missing: {wf}")
    return wf


@pytest.fixture(scope="session")
def wf_traverse_path() -> Path:
    wf = _workflow_path("KNIME_traverse_order")
    if not wf.exists():
        pytest.fail(f"Missing sample workflow: {wf}")
    return wf


# New: the functional test workflow with local paths
@pytest.fixture(scope="session")
def wf_knime_pp_2022_lr() -> Path:
    wf = _workflow_path("KNIME_PP_2022_LR")
    if not wf.exists():
        pytest.fail(f"Missing functional test workflow: {wf}")
    return wf


# --------------------------------------------------------------------
# Node-level test data fixtures (deduplicated)
# --------------------------------------------------------------------
@pytest.fixture(scope="session")
def node_csv_reader_dir() -> Path:
    ndir = DATA_DIR / "Node_csv_reader"
    settings = ndir / "settings.xml"
    if not settings.exists():
        pytest.fail(f"Missing CSV Reader node settings: {settings}")
    return ndir


# Back-compat alias (if older tests use this name)
@pytest.fixture(scope="session")
def csv_reader_node_dir(node_csv_reader_dir: Path) -> Path:
    return node_csv_reader_dir


@pytest.fixture(scope="session")
def node_csv_writer_dir() -> Path:
    p = DATA_DIR / "Node_csv_writer"
    settings = p / "settings.xml"
    if not settings.exists():
        pytest.fail(f"Missing writer node settings at {settings}")
    return p


@pytest.fixture(scope="session")
def node_column_filter_dir() -> Path:
    p = DATA_DIR / "Node_column_filter"
    settings = p / "settings.xml"
    if not settings.exists():
        pytest.fail(f"Missing Column Filter node settings at {settings}")
    return p


@pytest.fixture(scope="session")
def node_missing_value_dir() -> Path:
    p = DATA_DIR / "Node_missing_value"
    settings = p / "settings.xml"
    if not settings.exists():
        pytest.fail(f"Missing Missing Value node settings at {settings}")
    return p


@pytest.fixture(scope="session")
def node_normalizer_dir() -> Path:
    p = DATA_DIR / "Node_normalizer"
    settings = p / "settings.xml"
    if not settings.exists():
        pytest.fail(f"Missing Normalizer node settings: {settings}")
    return p


# --------------------------------------------------------------------
# Helpers for functional workflow tests
# --------------------------------------------------------------------
@pytest.fixture()
def generate_lr_workbook(k2p_script: Path, run_cli, clean_output_dir: Path):
    """
    Returns a function that runs k2p on the LR workflow and creates the Python workbook
    with graphs disabled, writing to tests/data/!output.
    """
    def _generate(workflow_knime: Path) -> Path:
        args = [
            str(workflow_knime.parent),   # CLI accepts project dir OR workflow.knime path
            "--out", str(clean_output_dir),
            "--graph", "off",
            "--workbook", "py",
        ]
        run_cli(k2p_script, args, cwd=REPO_ROOT, check=True)

        # Locate the single generated workbook *.py
        candidates = sorted(clean_output_dir.glob("*_workbook.py"))
        if not candidates:
            raise AssertionError(f"No *_workbook.py generated in {clean_output_dir}")
        if len(candidates) > 1:
            # If multiple components, pick the expected LR one or just take the first deterministically
            expected = clean_output_dir / "KNIME_PP_2022_LR__g01_workbook.py"
            return expected if expected.exists() else candidates[0]
        return candidates[0]

    return _generate


@pytest.fixture()
def run_python_script(python_exe: str):
    """Execute a Python script and fail test on non-zero exit."""
    def _run(script_path: Path, cwd: Optional[Path] = None):
        proc = subprocess.run(
            [python_exe, str(script_path)],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"Script failed ({proc.returncode}): {script_path}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )
        return proc
    return _run

@pytest.fixture()
def empty_output_dir() -> Path:
    """
    Ensure tests/data/!output exists and is empty BEFORE the test runs.
    Use this fixture in functional tests that write to that directory.
    """
    outdir = DATA_DIR / "!output"
    outdir.mkdir(parents=True, exist_ok=True)

    # Remove everything inside the directory (files, symlinks, subdirs)
    for p in outdir.iterdir():
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        except Exception as e:
            pytest.fail(f"Failed to clean output path {p}: {e}")

    return outdir