from pathlib import Path
import sys
import pytest

# Ensure repo root is on sys.path so `import knime2py...` works
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

DATA_DIR = Path(__file__).resolve().parent / "data"

def _workflow_path(project_dirname: str) -> Path:
    """tests/data/<project>/workflow.knime"""
    return DATA_DIR / project_dirname / "workflow.knime"

@pytest.fixture(scope="session")
def wf_single_csv_path() -> Path:
    wf = _workflow_path("KNIME_project_single_csv")
    if not wf.exists():
        pytest.fail(f"Missing sample workflow: {wf}")
    return wf

@pytest.fixture(scope="session")
def wf_io_csv_path() -> Path:
    wf = _workflow_path("KNIME_project_io_csv")
    if not wf.exists():
        pytest.fail(f"Missing sample workflow: {wf}")
    return wf
