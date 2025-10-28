# tests/test_csv_reader_node.py
from pathlib import Path

import pytest
import knime2py.nodes.csv_reader as cr
from knime2py.nodes.node_utils import looks_like_path


# Local fixtures: derive the node path from the generic `node_dir`,
# and provide a back-compat alias `csv_reader_node_dir`.
@pytest.fixture(scope="session")
def node_csv_reader_dir(node_dir) -> Path:
    """Fixture that provides the path to the CSV reader node directory."""
    return node_dir("Node_csv_reader")

@pytest.fixture(scope="session")
def csv_reader_node_dir(node_csv_reader_dir: Path) -> Path:
    """Fixture that provides a back-compat alias for the CSV reader node directory."""
    return node_csv_reader_dir


def test_parse_csv_reader_settings_fields_valid(csv_reader_node_dir: Path):
    """Ensure parse_csv_reader_settings fills all fields plausibly and correctly.

    This test verifies that the settings parsed from the CSV reader node
    contain valid values for path, delimiter, quote character, escape character,
    header, and encoding.
    """
    s = cr.parse_csv_reader_settings(csv_reader_node_dir)

    # path
    assert s.path, "Expected a path parsed from settings.xml"
    assert looks_like_path(s.path), f"Unexpected path format: {s.path!r}"

    # delimiter
    assert s.sep is not None
    assert isinstance(s.sep, str) and len(s.sep) >= 1

    # quotechar
    assert s.quotechar in {'"', "'"}, f"Unexpected quotechar: {s.quotechar!r}"

    # escapechar can be None or 1-char
    assert (s.escapechar is None) or (isinstance(s.escapechar, str) and len(s.escapechar) == 1)

    # header is a bool
    assert isinstance(s.header, bool)

    # encoding is a non-empty string
    assert isinstance(s.encoding, str) and s.encoding
