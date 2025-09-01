# tests/test_traverse_order.py
from pathlib import Path
import pytest
import knime2py.parse_knime as k2p
from knime2py.emitters import depth_order


def test_depth_ready_order_for_sample():
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
