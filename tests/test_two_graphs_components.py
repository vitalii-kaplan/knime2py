import re
from pathlib import Path
import knime2py.parse_knime as k2p

import pytest
import knime2py.parse_knime as k2p


@pytest.fixture(scope="session")
def wf_two_graphs_path(workflow) -> Path:
    return workflow("KNIME_two_graphs")

def test_two_graphs_discovery_includes_sample(wf_two_graphs_path: Path):
    root = wf_two_graphs_path.parent
    found = k2p.discover_workflows(root)
    assert any(p.samefile(wf_two_graphs_path) for p in found), "Sample workflow not discovered"


def test_two_graphs_components_structure(wf_two_graphs_path: Path):
    graphs = k2p.parse_workflow_components(wf_two_graphs_path)
    assert len(graphs) == 2, f"expected 2 components, got {len(graphs)}"

    # Component IDs should be suffixed __g01, __g02 with base = folder name
    base = wf_two_graphs_path.parent.name
    ids = {g.workflow_id for g in graphs}
    assert ids == {f"{base}__g01", f"{base}__g02"}

    # Each component: exactly 2 nodes, 1 edge, and the edge connects those two nodes
    def label(n):
        return f"{n.type or ''} {n.name or ''}"

    for g in graphs:
        assert len(g.nodes) == 2, f"{g.workflow_id}: nodes={len(g.nodes)}"
        assert len(g.edges) == 1, f"{g.workflow_id}: edges={len(g.edges)}"

        e = g.edges[0]
        assert e.source in g.nodes and e.target in g.nodes, f"{g.workflow_id}: edge connects unknown nodes"

        # One CSV Reader and one CSV Writer per component
        readers = [n for n in g.nodes.values()
                   if re.search(r"\bcsv\b", label(n), re.I) and re.search(r"\breader\b", label(n), re.I)]
        writers = [n for n in g.nodes.values()
                   if re.search(r"\bcsv\b", label(n), re.I) and re.search(r"\bwriter\b", label(n), re.I)]
        assert len(readers) == 1, f"{g.workflow_id}: expected 1 CSV Reader, got {len(readers)}"
        assert len(writers) == 1, f"{g.workflow_id}: expected 1 CSV Writer, got {len(writers)}"

        # Edge should be from reader -> writer
        reader_id = next(nid for nid, n in g.nodes.items() if n in readers)
        writer_id = next(nid for nid, n in g.nodes.items() if n in writers)
        assert e.source == reader_id and e.target == writer_id, \
            f"{g.workflow_id}: expected edge {reader_id}->{writer_id}, got {e.source}->{e.target}"

    # Components must be disjoint by node IDs
    comp_nodes = [set(g.nodes.keys()) for g in graphs]
    assert comp_nodes[0].isdisjoint(comp_nodes[1]), "Components share node IDs (should be disjoint)"
