"""Ensure every registered node handler responds to stubs consistently."""

import sys
from pathlib import Path
from types import SimpleNamespace


repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import registry


def _edge_stub(source: str, target: str, source_port: str, target_port: str) -> SimpleNamespace:
    """Return a namespaced object that mimics the parse_knime.Edge API."""
    return SimpleNamespace(
        source=source,
        target=target,
        source_port=source_port,
        target_port=target_port,
    )


def test_all_handlers_honor_stub_ports(tmp_path: Path):
    """
    Discover all handlers and ensure their emitted bodies reference the provided stubs.

    Every handler should incorporate at least one incoming context key (when applicable)
    and publish results under the outgoing port ids supplied by the stubs.
    """
    handlers = registry.get_handlers()
    fake_node_id = "HANDLER_STUB"

    modules_without_inputs = {"csv_reader", "db_connector", "excel_reader", "pmml_reader", "model_reader", "not_implemented"}
    modules_without_outputs = {"csv_writer", "excel_writer", "table_view", "roc_curve", "not_implemented"}

    failures: list[str] = []
    for factory_id, mod in handlers.items():
        module_name = mod.__name__.rsplit(".", 1)[-1]
        node_dir = tmp_path / module_name
        node_dir.mkdir(parents=True, exist_ok=True)

        incoming_edges = [
            ("SRC_A", _edge_stub("SRC_A", fake_node_id, "11", "1")),
            ("SRC_B", _edge_stub("SRC_B", fake_node_id, "22", "2")),
        ]
        outgoing_edges = [
            ("OUT_X", _edge_stub(fake_node_id, "OUT_X", "1", "1")),
            ("OUT_Y", _edge_stub(fake_node_id, "OUT_Y", "2", "2")),
        ]

        try:
            _, body = mod.handle(
                f"fake::{factory_id}",
                fake_node_id,
                str(node_dir),
                incoming_edges,
                outgoing_edges,
            )
        except Exception as exc:
            failures.append(f"{module_name}: handle() raised {exc}")
            continue

        body_text = "\n".join(body or [])

        if module_name not in modules_without_outputs:
            out_ok = fake_node_id in body_text
            if not out_ok:
                failures.append(f"{module_name}: body missing node id in output: {body_text}")

        if module_name not in modules_without_inputs:
            in_ok = any(f"{src}:{edge.source_port}" in body_text for src, edge in incoming_edges)
            if not in_ok:
                failures.append(f"{module_name}: body missing incoming stub reference: {body_text}")

    assert not failures, "Handlers failing stub propagation:\n" + "\n".join(failures)


def test_all_handlers_expose_get_name() -> None:
    """Every discovered handler should expose a callable get_name() -> non-empty str."""
    handlers = registry.get_handlers()
    failures: list[str] = []

    for _factory_id, mod in handlers.items():
        name_fn = getattr(mod, "get_name", None)
        if not callable(name_fn):
            failures.append(f"{mod.__name__}: missing callable get_name()")
            continue
        try:
            value = name_fn()
        except Exception as exc:
            failures.append(f"{mod.__name__}: get_name() raised {exc}")
            continue
        if not isinstance(value, str) or not value.strip():
            failures.append(f"{mod.__name__}: get_name() returned invalid value {value!r}")

    assert not failures, "Handlers failing get_name() contract:\n" + "\n".join(failures)
