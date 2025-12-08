"""Unit tests for the Color Manager node handler."""

import sys
from pathlib import Path
from types import SimpleNamespace

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import color_manager


def test_color_manager_handle_uses_incoming_and_outgoing_stubs(tmp_path: Path):
    """Verify that handle() wires context keys derived from incoming/outgoing stubs."""
    incoming = [("123", SimpleNamespace(source_port="7"))]
    outgoing = [("outA", SimpleNamespace(source_port="2")), ("outB", SimpleNamespace(source_port="5"))]

    imports, body = color_manager.handle("color", "999", tmp_path, incoming, outgoing)

    assert not imports, "Color Manager should not inject extra imports"
    assert "df = context['123:7']" in body
    for port in {"2", "5"}:
        line = f"context['999:{port}'] = df"
        assert line in body, f"Missing passthrough for port {port}\nBody:\n{body}"
