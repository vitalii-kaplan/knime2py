# tests/test_csv_reader_node.py
import re
from pathlib import Path

import pytest

from knime2py.nodes import csv_reader as cr


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def csv_reader_node_dir() -> Path:
    node_dir = _data_dir() / "Node_csv_reader"
    settings = node_dir / "settings.xml"
    if not settings.exists():
        pytest.skip(f"CSV Reader test settings not found at {settings}")
    return node_dir


def test_parse_csv_reader_settings_fields_valid(csv_reader_node_dir: Path):
    """Ensure parse_csv_reader_settings fills all fields plausibly and correctly."""
    s = cr.parse_csv_reader_settings(csv_reader_node_dir)

    # path
    assert s.path, "Expected a path parsed from settings.xml"
    assert cr._looks_like_path(s.path), f"Unexpected path format: {s.path!r}"

    # delimiter
    assert isinstance(s.sep, str) and len(s.sep) == 1, f"sep must be a single character, got {s.sep!r}"

    # quotechar
    assert s.quotechar is None or (isinstance(s.quotechar, str) and len(s.quotechar) == 1), \
        f"quotechar must be None or single char, got {s.quotechar!r}"

    # escapechar
    assert s.escapechar is None or (isinstance(s.escapechar, str) and len(s.escapechar) == 1), \
        f"escapechar must be None or single char, got {s.escapechar!r}"

    # header
    assert isinstance(s.header, bool), f"header must be bool, got {type(s.header)}"

    # encoding
    assert isinstance(s.encoding, str) and s.encoding, "encoding must be a non-empty string"
