# tests/test_nodeblocks_partitioning.py
import sys
from pathlib import Path

# Make package importable from repo root
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
from knime2py.parse_knime import WorkflowGraph, Node, Edge
from knime2py.emitters import build_workbook_blocks


def test_partitioning_block_emits_stratified_train_lines(node_csv_reader_dir: Path):
    """
    Build a minimal graph: Reader(1393) -> Partitioning(4001),
    then verify the Partitioning NodeBlock contains the expected
    stratified-train code based on tests/data/Node_partitioning/settings.xml.
    """
    node_partitioning_dir = repo_root / "tests" / "data" / "Node_partitioning"
    assert node_partitioning_dir.joinpath("settings.xml").exists(), "Missing Partitioning settings.xml test data"

    # Node ids & types
    reader_id = "1393"
    part_id = "4001"
    reader_type = "org.knime.base.node.io.filehandling.csv.reader.CSVTableReaderNodeFactory"
    part_type = "org.knime.base.node.preproc.partition.PartitionNodeFactory"

    # Minimal workflow graph
    g = WorkflowGraph(
        workflow_id="test_partitioning_block",
        workflow_path=str((repo_root / "tests" / "data" / "dummy" / "workflow.knime").resolve()),
        nodes={
            reader_id: Node(id=reader_id, name="CSV Reader", type=reader_type, path=str(node_csv_reader_dir)),
            part_id: Node(id=part_id, name="Partitioning", type=part_type, path=str(node_partitioning_dir)),
        },
        edges=[Edge(source=reader_id, target=part_id, source_port="1", target_port="1")],
    )

    blocks, _imports = build_workbook_blocks(g)
    assert blocks, "Expected NodeBlocks to be created"

    # Find the Partitioning block
    p_block = next((b for b in blocks if b.nid == part_id), None)
    assert p_block is not None, "Partitioning NodeBlock not found"

    code = "\n".join(p_block.code_lines)

    # Expected exact lines from the generator (stratified, 70/30, seed=1, class column 'Dependent')
    assert "_seed = 1" in code
    assert "_frac = 0.7" in code
    assert "parts = [g.sample(frac=_frac, random_state=_seed) for _, g in df.groupby('Dependent', dropna=False, sort=False)]" in code
    assert "train_df = pd.concat(parts, axis=0).sort_index() if parts else df.iloc[0:0]" in code
