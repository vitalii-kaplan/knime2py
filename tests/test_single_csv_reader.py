# tests/parse/test_discovery_and_parse.py

"""Test suite for parsing and discovering KNIME workflows.

Overview
----------------------------
This module contains tests for discovering and parsing KNIME workflows, specifically
focusing on the single CSV reader node.

Runtime Behavior
----------------------------
Inputs:
- The module reads the path to a single CSV workflow.

Outputs:
- The tests assert the structure of the parsed workflow, including the number of nodes
  and edges, as well as the state of the nodes.

Key algorithms or mappings:
- The module verifies that the discovered workflows include the expected sample workflow
  and checks the properties of the nodes in the parsed graph.

Edge Cases
----------------------------
The code checks for the presence of nodes and edges, ensuring that the graph structure
is as expected. It also verifies that the node state is correctly parsed.

Generated Code Dependencies
----------------------------
The generated code may depend on external libraries such as pandas and pytest. These
dependencies are required for the generated code, not for this test module.

Usage
----------------------------
This module is typically invoked during testing of the knime2py package. It is used to
validate the functionality of the workflow parsing.

Node Identity
----------------------------
The tests focus on the single CSV reader node, ensuring that it is correctly identified
and parsed.

Configuration
----------------------------
The tests do not generate code based on settings.xml, but they validate the parsing
of the single CSV reader node.

Limitations
----------------------------
This module does not cover all possible KNIME nodes or workflows, focusing instead on
the single CSV reader.

References
----------------------------
For more information, refer to the KNIME documentation and the knime2py project.

"""

import re
from pathlib import Path

import pytest
import knime2py.parse_knime as k2p

@pytest.fixture(scope="session")
def wf_single_csv_path(workflow) -> Path:
    """Fixture that provides the path to the single CSV workflow for testing."""
    return workflow("KNIME_single_csv")

def test_discovery_includes_sample(wf_single_csv_path: Path):
    """Test that the discovery function includes the sample workflow in the found workflows."""
    root = wf_single_csv_path.parent
    found = k2p.discover_workflows(root)
    assert any(p.samefile(wf_single_csv_path) for p in found)

def test_parse_single_csv_reader_only_one_node(wf_single_csv_path: Path):
    """Test that parsing the single CSV reader workflow results in a graph with one node and no edges."""
    g = k2p.parse_workflow(wf_single_csv_path)
    assert len(g.nodes) == 1
    assert len(g.edges) == 0

    (nid, node), = g.nodes.items()
    assert nid.isdigit()
    assert node.name is None or isinstance(node.name, str)
    assert re.search(r"\bcsv\b", node.name or "", re.I) and re.search(r"\breader\b", node.name or "", re.I)
    assert node.type and re.search(r"csv.*reader", node.type, re.I)
    assert node.path

def test_node_state_idle_from_settings(wf_single_csv_path: Path):
    """Test that the node state is correctly parsed as 'IDLE' from the settings of the single CSV reader node."""
    g = k2p.parse_workflow(wf_single_csv_path)

    # sanity
    assert len(g.nodes) == 1
    assert len(g.edges) == 0

    # the only node
    (nid, node), = g.nodes.items()
    assert node.state == "IDLE"
