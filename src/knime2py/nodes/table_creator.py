#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, iter_entries, split_out_imports


FACTORY = "org.knime.base.node.io.tablecreator.TableCreator2NodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.io.tablecreator.TableCreator2NodeFactory"
)


@dataclass
class TableCreatorColumn:
    index: int
    name: str
    cell_class: str = "org.knime.core.data.def.StringCell"
    skip: bool = False
    missing_pattern: str = ""


@dataclass
class TableCreatorSettings:
    columns: List[TableCreatorColumn] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)


def _bool(raw: Optional[str], default: bool = False) -> bool:
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


def _convert_value(raw: Optional[str], column: TableCreatorColumn) -> Any:
    if raw is None:
        return None
    value = str(raw)
    if column.missing_pattern and value == column.missing_pattern:
        return None

    cell_class = column.cell_class.rsplit(".", 1)[-1]
    if cell_class in {"IntCell", "LongCell"}:
        try:
            return int(value)
        except ValueError:
            return None
    if cell_class in {"DoubleCell", "FloatCell"}:
        try:
            return float(value)
        except ValueError:
            return None
    if cell_class == "BooleanCell":
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return value


class TableCreator:
    @staticmethod
    def parse(node_dir: Optional[Path]) -> TableCreatorSettings:
        if not node_dir:
            return TableCreatorSettings()
        settings_path = node_dir / "settings.xml"
        if not settings_path.exists():
            return TableCreatorSettings()

        root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
        model = first_el(root, ".//*[local-name()='config' and @key='model']")
        if model is None:
            return TableCreatorSettings()

        column_props = first_el(model, "./*[local-name()='config' and @key='columnProperties']")
        columns: List[TableCreatorColumn] = []
        if column_props is not None:
            for cfg in column_props.xpath("./*[local-name()='config' and string(number(@key)) != 'NaN']"):
                idx = int(cfg.get("key") or "0")
                name = first(cfg, "./*[local-name()='entry' and @key='ColumnName']/@value") or f"Column {idx + 1}"
                cell_class = (
                    first(
                        cfg,
                        "./*[local-name()='config' and @key='ColumnClass']"
                        "/*[local-name()='entry' and @key='cell_class']/@value",
                    )
                    or "org.knime.core.data.def.StringCell"
                )
                columns.append(
                    TableCreatorColumn(
                        index=idx,
                        name=name,
                        cell_class=cell_class,
                        skip=_bool(first(cfg, "./*[local-name()='entry' and @key='SkipThisColumn']/@value"), False),
                        missing_pattern=first(cfg, "./*[local-name()='entry' and @key='MissValuePattern']/@value") or "",
                    )
                )
        columns.sort(key=lambda col: col.index)

        row_indices = [int(v) for v in _indexed_values(first_el(model, "./*[local-name()='config' and @key='rowIndices']"))]
        column_indices = [
            int(v) for v in _indexed_values(first_el(model, "./*[local-name()='config' and @key='columnIndices']"))
        ]
        values = _indexed_values(first_el(model, "./*[local-name()='config' and @key='values']"))

        if not columns and column_indices:
            columns = [TableCreatorColumn(index=i, name=f"Column {i + 1}") for i in range(max(column_indices) + 1)]

        by_index = {col.index: col for col in columns}
        row_count = (max(row_indices) + 1) if row_indices else 0
        col_count = (max([col.index for col in columns], default=-1) + 1) if columns else 0
        matrix: List[List[Any]] = [[None for _ in range(col_count)] for _ in range(row_count)]

        for row_idx, col_idx, value in zip(row_indices, column_indices, values):
            column = by_index.get(col_idx, TableCreatorColumn(index=col_idx, name=f"Column {col_idx + 1}"))
            if row_idx >= len(matrix):
                matrix.extend([[None for _ in range(col_count)] for _ in range(row_idx - len(matrix) + 1)])
            if col_idx >= col_count:
                for row in matrix:
                    row.extend([None for _ in range(col_idx - len(row) + 1)])
                col_count = col_idx + 1
            matrix[row_idx][col_idx] = _convert_value(value, column)

        kept_columns = [col for col in columns if not col.skip]
        kept_indices = [col.index for col in kept_columns]
        rows = [[row[idx] if idx < len(row) else None for idx in kept_indices] for row in matrix]
        return TableCreatorSettings(columns=kept_columns, rows=rows)

    @staticmethod
    def emit(settings: TableCreatorSettings) -> List[str]:
        column_names = [col.name for col in settings.columns]
        rows = settings.rows
        return [
            f"_table_creator_columns = {repr(column_names)}",
            f"_table_creator_rows = {repr(rows)}",
            "df = pd.DataFrame(_table_creator_rows, columns=_table_creator_columns)",
        ]


def parse_table_creator_settings(node_dir: Optional[Path]) -> TableCreatorSettings:
    return TableCreator.parse(node_dir)


def generate_imports() -> List[str]:
    return ["import pandas as pd"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_table_creator_settings(ndir)
    lines: List[str] = [f"# {HUB_URL}"]
    lines.extend(TableCreator.emit(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = df")
    return lines


def get_name() -> str:
    return "Table Creator"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, [], out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
