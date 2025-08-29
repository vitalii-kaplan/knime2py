import re
from pathlib import Path
import knime2py.parse_knime as k2p

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
