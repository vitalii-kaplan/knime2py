# tests/test_nodeblocks_csv_writer.py
import re
from pathlib import Path
import sys
import pytest

# Make package importable
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.parse_knime import WorkflowGraph, Node, Edge
from knime2py.emitters import build_workbook_blocks
from knime2py.nodes import csv_reader, csv_writer


def test_writer_nodeblock_uses_settings_and_correct_context_key(
    node_csv_reader_dir: Path,
    node_csv_writer_dir: Path,
):
    """
    Build a minimal graph: Reader(1393) -> Writer(1394),
    then verify the Writer NodeBlock consumes the right context key
    and writes to the path parsed from its settings.xml with correct kwargs.
    """
    reader_id = "1393"
    writer_id = "1394"
    reader_type = "org.knime.base.node.io.filehandling.csv.reader.CSVTableReaderNodeFactory"
    writer_type = "org.knime.base.node.io.filehandling.csv.writer.CSVWriter2NodeFactory"

    g = WorkflowGraph(
        workflow_id="test_writer_block",
        workflow_path=str((repo_root / "tests" / "data" / "dummy" / "workflow.knime").resolve()),
        nodes={
            reader_id: Node(id=reader_id, name="CSV Reader", type=reader_type, path=str(node_csv_reader_dir)),
            writer_id: Node(id=writer_id, name="CSV Writer", type=writer_type, path=str(node_csv_writer_dir)),
        },
        edges=[Edge(source=reader_id, target=writer_id, source_port="1", target_port="1")],
    )

    blocks, _ = build_workbook_blocks(g)
    assert blocks, "Expected at least two NodeBlocks (reader and writer)"

    writer_block = next((b for b in blocks if b.nid == writer_id), None)
    assert writer_block is not None, "Writer NodeBlock not found"

    code = "\n".join(writer_block.code_lines).strip()

    # 1) Pulls df from the upstream context key "1393:1"
    assert "df = context['1393:1']" in code

    # 2) Uses output path from tests/data/Node_csv_writer/settings.xml
    assert 'out_path = Path(r"/Users/vitaly/Downloads/Sales_data_2022_out.csv")' in code

    # 3) to_csv kwargs reflect settings
    assert re.search(r"to_csv\(", code)
    assert re.search(r"sep\s*=\s*','", code)
    # Simpler and robust: single-quoted double quote (repr('"') -> '"\'"\'')
    assert "quotechar='\"'" in code
    assert re.search(r"header\s*=\s*True", code)
    assert re.search(r"encoding\s*=\s*'utf-8'", code, re.I)
    assert re.search(r"index\s*=\s*False", code)

    # missing_value_pattern="" → na_rep=''
    assert re.search(r"na_rep\s*=\s*''", code), "Expected na_rep='' from settings"

    # Sanity: writer hub link comment present
    assert csv_writer.FACTORY in code
