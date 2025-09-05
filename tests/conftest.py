from pathlib import Path
import sys
import pytest

# Ensure repo root is on sys.path so `import knime2py...` works
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

DATA_DIR = Path(__file__).resolve().parent / "data"


def _workflow_path(project_dirname: str) -> Path:
    return DATA_DIR / project_dirname / "workflow.knime"

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
    wf = DATA_DIR / "KNIME_two_graphs" / "workflow.knime"
    if not wf.exists():
        pytest.fail(f"Test data missing: {wf}")
    return wf

@pytest.fixture(scope="session")
def wf_traverse_path() -> Path:
    wf = _workflow_path("KNIME_traverse_order")
    if not wf.exists():
        pytest.fail(f"Missing sample workflow: {wf}")
    return wf

# --- Node-level test data fixtures ---

@pytest.fixture(scope="session")
def node_csv_writer_dir() -> Path:
    p = DATA_DIR / "Node_csv_writer"
    settings = p / "settings.xml"
    if not settings.exists():
        pytest.fail(f"Missing writer node settings at {settings}")
    return p

# --- Node-level test data fixtures ---

@pytest.fixture(scope="session")
def node_csv_reader_dir() -> Path:
    ndir = DATA_DIR / "Node_csv_reader"
    settings = ndir / "settings.xml"
    if not settings.exists():
        pytest.fail(f"Missing CSV Reader node settings: {settings}")
    return ndir

# Back-compat alias so tests that expect `csv_reader_node_dir` keep working
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