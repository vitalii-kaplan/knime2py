# tests/test_nodeblocks_missing_value.py
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


def test_missing_value_block_imputes_integers_to_zero(node_csv_reader_dir: Path):
    """
    Build a minimal graph: Reader(1393) -> MissingValue(3001),
    then verify the Missing Value NodeBlock reads from the correct context
    and emits code that imputes integer-typed columns with 0, based on
    tests/data/Node_missing_value/settings.xml.
    """
    # Path to the Missing Value node settings
    node_missing_value_dir = repo_root / "tests" / "data" / "Node_missing_value"
    assert node_missing_value_dir.joinpath("settings.xml").exists(), "Missing Missing Value settings.xml test data"

    print("test_missing_value_block_imputes_integers_to_zero dir is: ", node_missing_value_dir)

    # Node ids & types
    reader_id = "1393"
    mv_id = "3001"
    reader_type = "org.knime.base.node.io.filehandling.csv.reader.CSVTableReaderNodeFactory"
    mv_type = "org.knime.base.node.preproc.pmml.missingval.compute.MissingValueHandlerNodeFactory"

    # Minimal workflow graph
    g = WorkflowGraph(
        workflow_id="test_missing_value_block",
        workflow_path=str((repo_root / "tests" / "data" / "dummy" / "workflow.knime").resolve()),
        nodes={
            reader_id: Node(id=reader_id, name="CSV Reader", type=reader_type, path=str(node_csv_reader_dir)),
            mv_id: Node(id=mv_id, name="Missing Value", type=mv_type, path=str(node_missing_value_dir)),
        },
        edges=[Edge(source=reader_id, target=mv_id, source_port="1", target_port="1")],
    )

    blocks = build_workbook_blocks(g)
    assert blocks, "Expected NodeBlocks to be created"

    # Find the Missing Value block
    mv_block = next((b for b in blocks if b.nid == mv_id), None)
    assert mv_block is not None, "Missing Value NodeBlock not found"

    code = "\n".join(mv_block.code_lines)

    # 1) Pulls df from the upstream context key "1393:1"
    assert "df = context['1393:1']" in code

    # 2) It should select integer columns in some reasonable way
    #    Accept either a select_dtypes(include=['Int...']) approach or an is_integer_dtype check.
    assert (
        ("select_dtypes" in code and re.search(r"include\s*=\s*\[?['\"]Int", code, re.I))
        or ("is_integer_dtype" in code)
    ), "Expected integer column selection in generated code"

    # 3) It should impute missing integers with 0 (FixedIntegerValueMissingCellHandlerFactory -> 0)
    assert re.search(r"\.fillna\(\s*0\s*\)", code), "Expected integer NA imputation to 0"
