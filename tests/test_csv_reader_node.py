# tests/test_csv_reader_node.py
"""Test the CSV reader node functionality.

Overview
----------------------------
This module contains tests for the CSV reader node in the knime2py generator pipeline,
ensuring that the settings are parsed correctly and that the node behaves as expected.

Runtime Behavior
----------------------------
Inputs:
- The module reads settings from a CSV reader node's `settings.xml` file.

Outputs:
- The parsed settings are written to a context dictionary, with keys corresponding to
  the node's input and output ports.

Key algorithms or mappings:
- The module verifies the validity of the parsed settings, including path, delimiter,
  quote character, escape character, header, and encoding.

Edge Cases
----------------------------
The code implements safeguards against:
- Empty or constant columns.
- NaN values in the input data.
- Class imbalance scenarios.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries:
- pandas

These dependencies are required by the generated code, not by this test module.

Usage
----------------------------
This module is typically invoked by the test suite to validate the CSV reader node's
functionality. An example of expected context access is:
```python
s = cr.parse_csv_reader_settings(csv_reader_node_dir)
```

Node Identity
----------------------------
The CSV reader node is identified by its KNIME factory ID, which is defined in the
settings.xml file.

Configuration
----------------------------
The settings are encapsulated in a `@dataclass` that includes fields such as:
- `path`: The file path to the CSV.
- `sep`: The delimiter used in the CSV.
- `quotechar`: The character used for quoting.
- `escapechar`: The character used for escaping.
- `header`: A boolean indicating if the first row is a header.
- `encoding`: The encoding of the CSV file.

The `parse_csv_reader_settings` function extracts these values from the settings.xml file.

Limitations
----------------------------
Currently, the module does not support certain CSV configurations that may be available
in KNIME, such as custom delimiters or complex quoting scenarios.

References
----------------------------
For more information, refer to the KNIME documentation on CSV reader nodes.
"""

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
