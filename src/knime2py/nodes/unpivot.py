#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first_el, iter_entries, normalize_in_ports, split_out_imports


FACTORY = "org.knime.base.node.preproc.unpivot2.Unpivot2NodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.unpivot2.Unpivot2NodeFactory"
)


@dataclass
class UnpivotSettings:
    value_cols: List[str] = field(default_factory=list)
    retained_cols: List[str] = field(default_factory=list)
    missing_values: bool = True


def _bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _indexed_values(parent: Optional[ET._Element]) -> List[str]:
    if parent is None:
        return []
    values: list[tuple[int, str]] = []
    for key, value in iter_entries(parent):
        if key.isdigit() and value is not None:
            values.append((int(key), value))
    return [value for _, value in sorted(values)]


class Unpivot:
    @staticmethod
    def parse(node_dir: Optional[Path]) -> UnpivotSettings:
        if not node_dir:
            return UnpivotSettings()
        settings_path = node_dir / "settings.xml"
        if not settings_path.exists():
            return UnpivotSettings()

        root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
        model = first_el(root, ".//*[local-name()='config' and @key='model']")
        if model is None:
            return UnpivotSettings()

        value_cols = _indexed_values(
            first_el(
                model,
                "./*[local-name()='config' and @key='value_columns']"
                "/*[local-name()='config' and @key='included_names']",
            )
        )
        retained_cols = _indexed_values(
            first_el(
                model,
                "./*[local-name()='config' and @key='retained_columns']"
                "/*[local-name()='config' and @key='included_names']",
            )
        )
        missing_raw = model.xpath("./*[local-name()='entry' and @key='missing-values']/@value")

        return UnpivotSettings(
            value_cols=value_cols,
            retained_cols=retained_cols,
            missing_values=_bool(missing_raw[0] if missing_raw else None, True),
        )

    @staticmethod
    def emit(settings: UnpivotSettings) -> List[str]:
        return [
            f"_value_cols = {repr(settings.value_cols)}",
            f"_retained_cols = {repr(settings.retained_cols)}",
            f"_missing_values = {repr(settings.missing_values)}",
            "_value_cols = [c for c in _value_cols if c in df.columns]",
            "_retained_cols = [c for c in _retained_cols if c in df.columns and c not in _value_cols]",
            "if not _value_cols:",
            "    out_df = df.copy()",
            "else:",
            "    _source = df.copy()",
            "    _row_id_col = '_k2p_RowID'",
            "    _row_order_col = '_k2p_RowOrder'",
            "    _source[_row_id_col] = [idx if str(idx).startswith('Row') else f'Row{idx}' for idx in _source.index]",
            "    _source[_row_order_col] = range(len(_source))",
            "    out_df = _source.melt(",
            "        id_vars=[_row_id_col, _row_order_col] + _retained_cols,",
            "        value_vars=_value_cols,",
            "        var_name='ColumnNames',",
            "        value_name='ColumnValues',",
            "    )",
            "    if not _missing_values:",
            "        out_df = out_df[out_df['ColumnValues'].notna()]",
            "    out_df = out_df.sort_values(_row_order_col, kind='mergesort')",
            "    out_df = out_df.rename(columns={_row_id_col: 'RowIDs'})",
            "    out_df = out_df.drop(columns=[_row_order_col])",
            "    out_df = out_df[['RowIDs', 'ColumnNames', 'ColumnValues'] + _retained_cols]",
        ]


def parse_unpivot_settings(node_dir: Optional[Path]) -> UnpivotSettings:
    return Unpivot.parse(node_dir)


def generate_imports() -> List[str]:
    return ["import pandas as pd"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_unpivot_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']  # input table",
    ]
    lines.extend(Unpivot.emit(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "Unpivot"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
