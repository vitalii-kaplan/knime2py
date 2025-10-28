# tests/test_traverse_order.py
from pathlib import Path
import pytest
import knime2py.parse_knime as k2p
from knime2py.traverse import depth_order

@pytest.fixture(scope="session")
def node_csv_reader_dir(node_dir):
    """Fixture that provides the directory containing the KNIME traverse order CSV files.

    Args:
        node_dir (function): A function that returns the directory path for the specified node.

    Returns:
        Path: The path to the directory containing the KNIME traverse order CSV files.
    """
    return node_dir("KNIME_traverse_orderr")

def test_depth_ready_order_for_sample():
    """Test the depth-first traversal order of nodes in a KNIME workflow.

    This test verifies that the nodes in the workflow are traversed in the expected depth-first order,
    based on their numeric IDs. It checks for the presence of all nodes and ensures that there are no
    duplicates in the traversal.

    The expected order is defined by the numeric IDs of the nodes, and the test will skip if the
    sample workflow file is missing.

    Raises:
        pytest.SkipException: If the sample workflow file does not exist.
    """
    wf = Path(__file__).resolve().parent / "data" / "KNIME_traverse_order" / "workflow.knime"
    if not wf.exists():
        pytest.skip(f"Missing sample workflow: {wf}")

    g = k2p.parse_workflow(wf)

    # Expected order by numeric node IDs only (names are ignored)
    expected = [
        "1", "1350", "1351", "1365", "1362", "1386",
        "1390", "1389", "1385", "1360", "1364", "1387", "1388",
    ]

    got = depth_order(g.nodes, g.edges)

    # Sanity: same set, no duplicates, same length
    assert set(got) == set(g.nodes.keys()), f"Traversal missed nodes. got={got}, nodes={sorted(g.nodes.keys())}"
    assert len(got) == len(g.nodes), "Traversal contains duplicates or skipped nodes"

    # Exact sequence match on ids
    assert got == expected, f"Depth-ready order mismatch.\nExpected: {expected}\nGot:      {got}"
