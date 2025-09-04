# tests/test_csv_reader_node.py
from pathlib import Path
import knime2py.nodes.csv_reader as cr
from knime2py.nodes.node_utils import looks_like_path  # public helper

def test_parse_csv_reader_settings_fields_valid(csv_reader_node_dir: Path):
    """Ensure parse_csv_reader_settings fills all fields plausibly and correctly."""
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
