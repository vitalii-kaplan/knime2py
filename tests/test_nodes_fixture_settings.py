"""Exercise each node handler against a real fixture settings.xml directory."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path
from types import SimpleNamespace


repo_root = Path(__file__).resolve().parents[1]
src_root = repo_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from knime2py import nodes as nodes_pkg  # noqa: E402


_SKIP_MODULES = {"node_utils", "normalizer_utils", "pmml_utils", "registry", "not_implemented"}


def _edge_stub(source: str, target: str, source_port: str, target_port: str) -> SimpleNamespace:
    """Return a namespaced object that mimics the parse_knime.Edge API."""
    return SimpleNamespace(
        source=source,
        target=target,
        source_port=source_port,
        target_port=target_port,
    )


def _iter_node_modules():
    """Yield concrete node handler modules from knime2py.nodes."""
    for spec in sorted(pkgutil.iter_modules(nodes_pkg.__path__), key=lambda s: s.name):
        if spec.ispkg or spec.name.startswith("_") or spec.name in _SKIP_MODULES:
            continue

        mod = importlib.import_module(f"{nodes_pkg.__name__}.{spec.name}")
        if callable(getattr(mod, "handle", None)):
            yield spec.name, mod


def _find_fixture_dir(node_name: str) -> Path | None:
    """Return the shortest fixture dir whose name starts with the node's get_name()."""
    fixture_root = repo_root / "tests" / "data" / "nodes"
    matches = [path for path in fixture_root.iterdir() if path.is_dir() and path.name.startswith(node_name)]
    if not matches:
        return None
    return sorted(matches, key=lambda path: (len(path.name), path.name))[0]


def test_all_node_handlers_accept_real_fixture_settings() -> None:
    """
    For every concrete node handler, locate the corresponding fixture directory in
    tests/data/nodes and ensure handle(...) can build code from that settings.xml.
    """
    incoming_edges = [
        ("SRC_A", _edge_stub("SRC_A", "NODE", "1", "1")),
        ("SRC_B", _edge_stub("SRC_B", "NODE", "2", "2")),
        ("SRC_C", _edge_stub("SRC_C", "NODE", "3", "3")),
    ]
    outgoing_edges = [
        ("OUT_A", _edge_stub("NODE", "OUT_A", "1", "1")),
        ("OUT_B", _edge_stub("NODE", "OUT_B", "2", "2")),
        ("OUT_C", _edge_stub("NODE", "OUT_C", "3", "3")),
    ]

    failures: list[str] = []

    for module_name, mod in _iter_node_modules():
        name_fn = getattr(mod, "get_name", None)
        if not callable(name_fn):
            failures.append(f"{module_name}: missing callable get_name()")
            continue

        try:
            node_name = name_fn()
        except Exception as exc:
            failures.append(f"{module_name}: get_name() raised {exc}")
            continue

        fixture_dir = _find_fixture_dir(node_name)
        if fixture_dir is None:
            failures.append(f"{module_name}: no fixture directory starts with {node_name!r}")
            continue

        settings_xml = fixture_dir / "settings.xml"
        if not settings_xml.exists():
            failures.append(f"{module_name}: missing settings.xml in {fixture_dir.name}")
            continue

        try:
            imports, body = mod.handle(
                f"fixture::{module_name}",
                "NODE",
                str(fixture_dir),
                incoming_edges,
                outgoing_edges,
            )
        except Exception as exc:
            failures.append(f"{module_name}: handle() raised for {fixture_dir.name}: {exc}")
            continue

        if not isinstance(imports, list):
            failures.append(f"{module_name}: imports is not a list for {fixture_dir.name}")
        if not isinstance(body, list):
            failures.append(f"{module_name}: body is not a list for {fixture_dir.name}")

    assert not failures, "Handlers failing real settings.xml fixture test:\n" + "\n".join(failures)
