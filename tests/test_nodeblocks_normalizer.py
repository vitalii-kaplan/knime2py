# tests/test_nodeblocks_normalizer.py
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

# Provide the CSV Reader node dir (used by many tests; override here to avoid conftest edits)
@pytest.fixture(scope="session")
def node_csv_reader_dir(node_dir: callable) -> Path:  # node_dir comes from conftest
    return node_dir("Node_csv_reader")

@pytest.fixture(scope="session")
def node_normalizer_dir(node_dir: callable) -> Path:
    return node_dir("Node_normalizer")


def test_normalizer_block_minmax_defaults(node_csv_reader_dir: Path, node_normalizer_dir: Path):
    """
    Build a minimal graph: Reader(1393) -> Normalizer(3101), then verify the Normalizer NodeBlock:
      1) reads df from context['1393:1']
      2) selects numeric/boolean columns via select_dtypes(include=['number','bool','boolean','Int64','Float64'])
      3) emits min-max normalization with 0.0..1.0 and uses _minmax_col via .apply(_minmax_col)
      4) publishes to context['3101:1']
    Based on tests/data/Node_normalizer/settings.xml (MINMAX, new-min=0.0, new-max=1.0).
    """
    # Node ids & types
    reader_id = "1393"
    norm_id = "3101"
    reader_type = "org.knime.base.node.io.filehandling.csv.reader.CSVTableReaderNodeFactory"
    norm_type = "org.knime.base.node.preproc.normalize3.Normalizer3NodeFactory"

    # Use an existing workflow.knime path to avoid dummy-path surprises
    wf_path = (repo_root / "tests" / "data" / "KNIME_io_csv" / "workflow.knime").resolve()

    # Minimal workflow graph
    g = WorkflowGraph(
        workflow_id="test_normalizer_block",
        workflow_path=str(wf_path),
        nodes={
            reader_id: Node(id=reader_id, name="CSV Reader", type=reader_type, path=str(node_csv_reader_dir)),
            norm_id:   Node(id=norm_id,   name="Normalizer", type=norm_type,   path=str(node_normalizer_dir)),
        },
        edges=[Edge(source=reader_id, target=norm_id, source_port="1", target_port="1")],
    )

    blocks, _ = build_workbook_blocks(g)
    assert blocks, "Expected NodeBlocks to be created"

    # Find the Normalizer block
    norm_block = next((b for b in blocks if getattr(b, "nid", None) == norm_id), None)
    assert norm_block is not None, "Normalizer NodeBlock not found"

    code = "\n".join(norm_block.code_lines)

    # 1) Pulls df from the upstream context key "1393:1"
    assert "df = context['1393:1']" in code

    # 2) Should select numeric/boolean columns with the expected include list
    #    We check both the call and a couple of key tokens to avoid overfitting to formatting.
    assert "select_dtypes" in code
    assert "include=['number', 'bool', 'boolean', 'Int64', 'Float64']" in code

    # 3) MINMAX with 0.0..1.0, and the helper function naming
    assert re.search(r"_new_min\s*,\s*_new_max\s*=\s*0\.0\s*,\s*1\.0", code), "Expected 0.0..1.0 min-max bounds"
    assert "def _minmax_col(" in code and "apply(_minmax_col)" in code

    # 4) Publishes to this node's context key (port 1 by default)
    assert f"context['{norm_id}:1'] = out_df" in code
