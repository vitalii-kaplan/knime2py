import re
from pathlib import Path
import knime2py.parse_knime as k2p


def test_io_csv_discovery_includes_sample(wf_io_csv_path: Path):
    root = wf_io_csv_path.parent
    found = k2p.discover_workflows(root)
    assert any(p.samefile(wf_io_csv_path) for p in found), f"did not find {wf_io_csv_path}"


def test_io_csv_has_reader_writer_and_edge(wf_io_csv_path: Path):
    g = k2p.parse_workflow(wf_io_csv_path)

    # exactly two nodes
    assert len(g.nodes) == 2, f"expected 2 nodes, got {len(g.nodes)}"

    # classify by factory type (preferred) or fallback to name
    def label(n):
        return (n.type or "") + " " + (n.name or "")

    readers = [(nid, n) for nid, n in g.nodes.items()
               if re.search(r"\bcsv\b", label(n), re.I) and re.search(r"\breader\b", label(n), re.I)]
    writers = [(nid, n) for nid, n in g.nodes.items()
               if re.search(r"\bcsv\b", label(n), re.I) and re.search(r"\bwriter\b", label(n), re.I)]

    assert len(readers) == 1, f"expected 1 CSV Reader, got {[(nid, label(n)) for nid, n in readers]}"
    assert len(writers) == 1, f"expected 1 CSV Writer, got {[(nid, label(n)) for nid, n in writers]}"

    reader_id = readers[0][0]
    writer_id = writers[0][0]

    # at least one edge from reader -> writer
    assert any(e.source == reader_id and e.target == writer_id for e in g.edges), \
        f"no edge from reader {reader_id} to writer {writer_id}; edges={[(e.source, e.target) for e in g.edges]}"


def test_io_csv_reader_state_executed(wf_io_csv_path: Path):
    """CSV Reader node should be in EXECUTED state in this sample."""
    g = k2p.parse_workflow(wf_io_csv_path)

    def label(n):
        return (n.type or "") + " " + (n.name or "")

    readers = [(nid, n) for nid, n in g.nodes.items()
               if re.search(r"\bcsv\b", label(n), re.I) and re.search(r"\breader\b", label(n), re.I)]
    assert len(readers) == 1, f"expected 1 CSV Reader, got {[(nid, label(n)) for nid, n in readers]}"

    _, reader = readers[0]
    assert reader.state == "EXECUTED", f"CSV Reader state expected EXECUTED, got {reader.state!r}"


def test_io_csv_writer_state_configured(wf_io_csv_path: Path):
    """CSV Writer node should be in CONFIGURED state in this sample."""
    g = k2p.parse_workflow(wf_io_csv_path)

    def label(n):
        return (n.type or "") + " " + (n.name or "")

    writers = [(nid, n) for nid, n in g.nodes.items()
               if re.search(r"\bcsv\b", label(n), re.I) and re.search(r"\bwriter\b", label(n), re.I)]
    assert len(writers) == 1, f"expected 1 CSV Writer, got {[(nid, label(n)) for nid, n in writers]}"

    _, writer = writers[0]
    assert writer.state == "CONFIGURED", f"CSV Writer state expected CONFIGURED, got {writer.state!r}"
