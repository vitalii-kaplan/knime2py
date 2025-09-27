# tests/parse/test_discovery_and_parse.py

import re
from pathlib import Path

import pytest
import knime2py.parse_knime as k2p

@pytest.fixture(scope="session")
def wf_single_csv_path(workflow) -> Path:
    return workflow("KNIME_single_csv")

def test_discovery_includes_sample(wf_single_csv_path: Path):
    root = wf_single_csv_path.parent
    found = k2p.discover_workflows(root)
    assert any(p.samefile(wf_single_csv_path) for p in found)


def test_parse_single_csv_reader_only_one_node(wf_single_csv_path: Path):
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
    """
    The single-CSV sample has one node whose settings.xml contains:
      <entry key="state" type="xstring" value="IDLE"/>
    Verify the parser surfaces that as node.state == "IDLE".
    """
    g = k2p.parse_workflow(wf_single_csv_path)

    # sanity
    assert len(g.nodes) == 1
    assert len(g.edges) == 0

    # the only node
    (nid, node), = g.nodes.items()
    assert node.state == "IDLE"
