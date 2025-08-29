import os
from pathlib import Path
import pytest
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def _resolve_sample_path() -> Path:
    env = os.environ.get("K2P_SAMPLE_SINGLE_CSV")
    if env:
        p = Path(env)
        if p.is_dir():
            wf = p / "workflow.knime"
            if wf.exists():
                return wf
        elif p.is_file() and p.name == "workflow.knime":
            return p
    return Path(__file__).resolve().parent / "data" / "KNIME_project_single_csv" / "workflow.knime"

@pytest.fixture(scope="session")
def wf_single_csv_path() -> Path:
    wf = _resolve_sample_path()
    if not wf.exists():
        pytest.skip(
            f"Missing sample workflow: {wf}. "
            "Set env K2P_SAMPLE_SINGLE_CSV to the sample project dir or workflow.knime."
        )
    return wf
