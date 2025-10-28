# tests/test_io_csv_workflow.py
"""Test suite for CSV I/O workflows in KNIME.

Overview
----------------------------
This module contains tests to verify the functionality of CSV input and output nodes
within KNIME workflows, ensuring that the expected nodes are present and in the correct
state.

Runtime Behavior
----------------------------
Inputs:
- The module reads the path to a KNIME workflow that includes CSV Reader and Writer nodes.

Outputs:
- The tests assert the presence and state of the CSV Reader and Writer nodes in the
  workflow, checking their execution and configuration states.

Key algorithms or mappings:
- The tests utilize regular expressions to identify nodes based on their labels.

Edge Cases
----------------------------
The tests check for the presence of exactly one CSV Reader and one CSV Writer, ensuring
that no additional nodes interfere with the expected workflow structure.

Generated Code Dependencies
----------------------------
The generated code depends on the following external libraries:
- pandas
- pytest
These dependencies are required for the generated code, not for this test module.

Usage
----------------------------
This module is typically invoked by the pytest framework to run the defined test cases.
An example of expected context access is:
```python
workflow("KNIME_io_csv")
```

Node Identity
----------------------------
The tests focus on nodes identified as CSV Reader and CSV Writer, ensuring they are
correctly labeled and functioning.

Configuration
----------------------------
No specific configuration is required for this test module.

Limitations
----------------------------
This module does not cover all possible edge cases or configurations of CSV nodes in
KNIME workflows.

References
----------------------------
Refer to the KNIME documentation for more information on CSV Reader and Writer nodes.
"""

import re
from pathlib import Path

import pytest
import knime2py.parse_knime as k2p


@pytest.fixture(scope="session")
def wf_io_csv_path(workflow) -> Path:
    """Fixture to resolve the path to the KNIME workflow for CSV I/O tests."""
    return workflow("KNIME_io_csv")


def test_io_csv_discovery_includes_sample(wf_io_csv_path: Path):
    """Test that the workflow discovery includes the sample CSV workflow."""
    root = wf_io_csv_path.parent
    found = k2p.discover_workflows(root)
    assert any(p.samefile(wf_io_csv_path) for p in found), f"did not find {wf_io_csv_path}"


def test_io_csv_has_reader_writer_and_edge(wf_io_csv_path: Path):
    """Test that the workflow has exactly one CSV Reader and one CSV Writer, and an edge between them."""
    g = k2p.parse_workflow(wf_io_csv_path)

    # exactly two nodes
    assert len(g.nodes) == 2, f"expected 2 nodes, got {len(g.nodes)}"

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
    """Test that the CSV Reader node is in EXECUTED state in this sample."""
    g = k2p.parse_workflow(wf_io_csv_path)

    def label(n):
        return (n.type or "") + " " + (n.name or "")

    readers = [(nid, n) for nid, n in g.nodes.items()
               if re.search(r"\bcsv\b", label(n), re.I) and re.search(r"\breader\b", label(n), re.I)]
    assert len(readers) == 1, f"expected 1 CSV Reader, got {[(nid, label(n)) for nid, n in readers]}"

    _, reader = readers[0]
    assert reader.state == "EXECUTED", f"CSV Reader state expected EXECUTED, got {reader.state!r}"


def test_io_csv_writer_state_configured(wf_io_csv_path: Path):
    """Test that the CSV Writer node is in CONFIGURED state in this sample."""
    g = k2p.parse_workflow(wf_io_csv_path)

    def label(n):
        return (n.type or "") + " " + (n.name or "")

    writers = [(nid, n) for nid, n in g.nodes.items()
               if re.search(r"\bcsv\b", label(n), re.I) and re.search(r"\bwriter\b", label(n), re.I)]
    assert len(writers) == 1, f"expected 1 CSV Writer, got {[(nid, label(n)) for nid, n in writers]}"

    _, writer = writers[0]
    assert writer.state == "CONFIGURED", f"CSV Writer state expected CONFIGURED, got {writer.state!r}"

