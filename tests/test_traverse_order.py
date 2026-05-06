# tests/test_traverse_order.py
"""
Test the depth-first traversal order of nodes in a KNIME workflow.

Overview
----------------------------
This module tests the depth-first traversal order of nodes in a KNIME workflow,
ensuring that nodes are traversed in the expected order based on their numeric IDs.

Runtime Behavior
----------------------------
Inputs:
- The module reads a KNIME workflow file, which contains nodes and edges.

Outputs:
- The test verifies the order of node traversal and checks for duplicates.

Key algorithms:
- The depth-first traversal is implemented using the `depth_order` function.

Edge Cases
----------------------------
The code checks for the existence of the workflow file and skips the test if it is missing.

Generated Code Dependencies
----------------------------
This module does not generate code directly, but the generated code may depend on
external libraries such as pandas, numpy, and others.

Usage
----------------------------
This module is typically invoked by the test suite to validate the traversal order.
Example context access:
- `g = k2p.parse_workflow(wf)`

Node Identity
----------------------------
This module does not generate code based on `settings.xml`.

Configuration
----------------------------
This module does not generate code based on `settings.xml`.

Limitations
----------------------------
No options are currently unsupported.

References
----------------------------
Refer to the KNIME documentation for more information on workflow structures.
"""

from pathlib import Path
from types import SimpleNamespace
import pytest
import knime2py.parse_knime as k2p
from knime2py.parse_knime import Edge
from knime2py.traverse import (
    X_AGGREGATOR_FACTORY,
    X_PARTITIONER_FACTORY,
    depth_order,
)

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

    # Expected dependency-ready order with numeric tie-breaking (names are ignored)
    expected = [
        "1", "1350", "1351", "1365", "1362", "1386",
        "1387", "1388", "1390", "1389", "1385", "1360", "1364",
    ]

    got = depth_order(g.nodes, g.edges)

    # Sanity: same set, no duplicates, same length
    assert set(got) == set(g.nodes.keys()), f"Traversal missed nodes. got={got}, nodes={sorted(g.nodes.keys())}"
    assert len(got) == len(g.nodes), "Traversal contains duplicates or skipped nodes"

    # Exact sequence match on ids
    assert got == expected, f"Depth-ready order mismatch.\nExpected: {expected}\nGot:      {got}"


def test_depth_order_emits_all_predecessors_before_joined_consumer():
    """
    Regression for a diamond graph where DFS entered one joiner before all roots
    were visited, then emitted its consumer while that joiner was still on-stack.
    """
    nodes = {nid: object() for nid in ["1545", "1587", "1546", "1547", "1551", "1591", "1592", "1556"]}
    edges = [
        Edge("1545", "1547", "1", "2"),
        Edge("1587", "1547", "1", "1"),
        Edge("1546", "1551", "1", "2"),
        Edge("1587", "1551", "1", "1"),
        Edge("1591", "1592", "1", "2"),
        Edge("1587", "1592", "1", "1"),
        Edge("1547", "1556", "1", "1"),
        Edge("1551", "1556", "1", "2"),
        Edge("1592", "1556", "1", "3"),
    ]

    got = depth_order(nodes, edges)
    pos = {nid: idx for idx, nid in enumerate(got)}

    assert pos["1547"] < pos["1556"]
    assert pos["1551"] < pos["1556"]
    assert pos["1592"] < pos["1556"]


def test_depth_order_keeps_parallel_xvalidation_regions_contiguous():
    """
    X-Partitioner/X-Aggregator pairs are structured control-flow regions.

    A plain topological order may start the next partitioner before the current
    aggregator, which makes the emitter nest independent Python loops.
    """
    nodes = {
        "90": SimpleNamespace(type="reader"),
        "60": SimpleNamespace(type=X_PARTITIONER_FACTORY),
        "33": SimpleNamespace(type="sampler"),
        "29": SimpleNamespace(type="learner"),
        "31": SimpleNamespace(type="predictor"),
        "62": SimpleNamespace(type=X_AGGREGATOR_FACTORY),
        "61": SimpleNamespace(type=X_PARTITIONER_FACTORY),
        "12": SimpleNamespace(type="sampler"),
        "9": SimpleNamespace(type="learner"),
        "10": SimpleNamespace(type="predictor"),
        "66": SimpleNamespace(type=X_AGGREGATOR_FACTORY),
    }
    edges = [
        Edge("90", "60", "1", "1"),
        Edge("60", "33", "1", "1"),
        Edge("33", "29", "1", "1"),
        Edge("29", "31", "1", "1"),
        Edge("60", "31", "2", "2"),
        Edge("31", "62", "1", "1"),
        Edge("90", "61", "1", "1"),
        Edge("61", "12", "1", "1"),
        Edge("12", "9", "1", "1"),
        Edge("9", "10", "1", "1"),
        Edge("61", "10", "2", "2"),
        Edge("10", "66", "1", "1"),
    ]

    got = depth_order(nodes, edges)
    pos = {nid: idx for idx, nid in enumerate(got)}

    assert pos["60"] < pos["33"] < pos["29"] < pos["31"] < pos["62"]
    assert pos["62"] < pos["61"]
    assert pos["61"] < pos["12"] < pos["9"] < pos["10"] < pos["66"]
