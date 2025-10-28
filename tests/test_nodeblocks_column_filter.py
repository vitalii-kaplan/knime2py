# tests/test_nodeblocks_column_filter.py
import re
import sys
from pathlib import Path

# Make package importable from repo root
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
from knime2py.parse_knime import WorkflowGraph, Node, Edge
from knime2py.emitters import build_workbook_blocks

@pytest.fixture(scope="session")
def node_column_filter_dir(node_dir):
    """
    Fixture that provides the directory path for the Node_column_filter test data.

    Args:
        node_dir (function): A function that returns the directory path for a given node.

    Returns:
        Path: The path to the Node_column_filter directory.
    """
    return node_dir("Node_column_filter")

def test_column_filter_block_emits_expected_excludes_line(node_csv_reader_dir: Path):
    """
    Test that the Column Filter NodeBlock emits the expected exclude_cols line.

    This test builds a minimal workflow graph consisting of a CSV Reader node and a Column Filter node.
    It verifies that the generated NodeBlock for the Column Filter contains the expected list of columns
    to exclude based on the settings defined in the corresponding settings.xml file.

    Args:
        node_csv_reader_dir (Path): The directory path for the CSV Reader node.
    """
    # Paths
    node_column_filter_dir = repo_root / "tests" / "data" / "Node_column_filter"
    assert node_column_filter_dir.joinpath("settings.xml").exists(), "Missing writer settings.xml test data"

    # Node ids & types
    reader_id = "1393"
    filter_id = "2001"
    reader_type = "org.knime.base.node.io.filehandling.csv.reader.CSVTableReaderNodeFactory"
    filter_type = "org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory"

    # Minimal workflow graph
    g = WorkflowGraph(
        workflow_id="test_column_filter_block",
        workflow_path=str((repo_root / "tests" / "data" / "dummy" / "workflow.knime").resolve()),
        nodes={
            reader_id: Node(id=reader_id, name="CSV Reader", type=reader_type, path=str(node_csv_reader_dir)),
            filter_id: Node(id=filter_id, name="Column Filter", type=filter_type, path=str(node_column_filter_dir)),
        },
        edges=[Edge(source=reader_id, target=filter_id, source_port="1", target_port="1")],
    )

    blocks, _ = build_workbook_blocks(g)
    assert blocks, "Expected NodeBlocks to be created"

    # Find the Column Filter block
    cf_block = next((b for b in blocks if b.nid == filter_id), None)
    assert cf_block is not None, "Column Filter NodeBlock not found"

    code = "\n".join(cf_block.code_lines)

    # Exact line expected from tests/data/Node_column_filter/settings.xml
    expected_line = (
        "exclude_cols = ['uc_user', 'uc_course', 'uc_created', 'as_time', "
        "'ol_first_time', 'in_time_pay', 'ir_time']"
    )
    assert expected_line in code, f"Missing expected exclude list line.\nCode was:\n{code}"
