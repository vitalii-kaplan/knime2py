#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first_el, iter_entries, normalize_in_ports, split_out_imports


FACTORY = "org.knime.base.node.preproc.columnresorter.ColumnResorterNodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.columnresorter.ColumnResorterNodeFactory"
)

UNKNOWN_COLUMN_PLACEHOLDER = "<any unknown new column>"


@dataclass
class ColumnResorterSettings:
    column_order: List[str] = field(default_factory=list)


def _ordered_entries(cfg: ET._Element) -> List[str]:
    numbered: List[tuple[int, str]] = []
    for key, value in iter_entries(cfg):
        if key.isdigit() and value:
            numbered.append((int(key), value))
    return [value for _, value in sorted(numbered, key=lambda item: item[0])]


def parse_column_resorter_settings(node_dir: Optional[Path]) -> ColumnResorterSettings:
    if not node_dir:
        return ColumnResorterSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return ColumnResorterSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return ColumnResorterSettings()

    order_cfg = first_el(model, "./*[local-name()='config' and @key='ColumnOrder']")
    if order_cfg is None:
        order_cfg = first_el(model, ".//*[local-name()='config' and @key='ColumnOrder']")
    if order_cfg is None:
        return ColumnResorterSettings()

    return ColumnResorterSettings(column_order=_ordered_entries(order_cfg))


class ColumnResorter:
    @staticmethod
    def emit(settings: ColumnResorterSettings) -> List[str]:
        return [
            f"_column_order = {repr(settings.column_order)}",
            f"_unknown_placeholder = {repr(UNKNOWN_COLUMN_PLACEHOLDER)}",
            "_seen = set()",
            "_explicit = []",
            "for _col in _column_order:",
            "    if _col == _unknown_placeholder:",
            "        continue",
            "    if _col in df.columns and _col not in _seen:",
            "        _explicit.append(_col)",
            "        _seen.add(_col)",
            "_remaining = [col for col in df.columns if col not in _seen]",
            "if _unknown_placeholder in _column_order:",
            "    _out_columns = []",
            "    _remaining_inserted = False",
            "    for _col in _column_order:",
            "        if _col == _unknown_placeholder:",
            "            if not _remaining_inserted:",
            "                _out_columns.extend(_remaining)",
            "                _remaining_inserted = True",
            "        elif _col in df.columns and _col not in _out_columns:",
            "            _out_columns.append(_col)",
            "    if not _remaining_inserted:",
            "        _out_columns.extend(_remaining)",
            "else:",
            "    _out_columns = _explicit + _remaining",
            "out_df = df.loc[:, _out_columns].copy()",
        ]


def generate_imports() -> List[str]:
    return ["import pandas as pd"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_column_resorter_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']  # input table",
    ]
    lines.extend(ColumnResorter.emit(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "Column Resorter"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
