#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, iter_entries, normalize_in_ports, split_out_imports


FACTORY = "org.knime.base.node.preproc.pivot.Pivot2NodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.pivot.Pivot2NodeFactory"
)


@dataclass
class PivotAggregation:
    column: str
    method: str
    include_missing: bool = False


@dataclass
class PivotSettings:
    group_cols: List[str] = field(default_factory=list)
    pivot_cols: List[str] = field(default_factory=list)
    aggregations: List[PivotAggregation] = field(default_factory=list)
    missing_values: bool = True
    total_aggregation: bool = False
    retain_order: bool = False
    sort_lexicographical: bool = False


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


class Pivot:
    @staticmethod
    def parse(node_dir: Optional[Path]) -> PivotSettings:
        if not node_dir:
            return PivotSettings()
        settings_path = node_dir / "settings.xml"
        if not settings_path.exists():
            return PivotSettings()

        root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
        model = first_el(root, ".//*[local-name()='config' and @key='model']")
        if model is None:
            return PivotSettings()

        group_cols = _indexed_values(
            first_el(
                model,
                "./*[local-name()='config' and @key='grouByColumns']"
                "/*[local-name()='config' and @key='InclList']",
            )
        )
        pivot_cols = _indexed_values(
            first_el(
                model,
                "./*[local-name()='config' and @key='pivotColumns']"
                "/*[local-name()='config' and @key='InclList']",
            )
        )

        aggregation_el = first_el(model, "./*[local-name()='config' and @key='aggregationColumn']")
        columns = _indexed_values(
            first_el(aggregation_el, "./*[local-name()='config' and @key='columnNames']")
            if aggregation_el is not None
            else None
        )
        methods = _indexed_values(
            first_el(aggregation_el, "./*[local-name()='config' and @key='aggregationMethod']")
            if aggregation_el is not None
            else None
        )
        missing = _indexed_values(
            first_el(aggregation_el, "./*[local-name()='config' and @key='inclMissingVals']")
            if aggregation_el is not None
            else None
        )

        aggregations: List[PivotAggregation] = []
        for idx, column in enumerate(columns):
            if not column:
                continue
            method = methods[idx] if idx < len(methods) and methods[idx] else "Count"
            include_missing = _bool(missing[idx], False) if idx < len(missing) else False
            aggregations.append(PivotAggregation(column=column, method=method, include_missing=include_missing))

        return PivotSettings(
            group_cols=group_cols,
            pivot_cols=pivot_cols,
            aggregations=aggregations,
            missing_values=_bool(first(model, "./*[local-name()='entry' and @key='missing_values']/@value"), True),
            total_aggregation=_bool(first(model, "./*[local-name()='entry' and @key='total_aggregation']/@value"), False),
            retain_order=_bool(first(model, "./*[local-name()='entry' and @key='retainOrder']/@value"), False),
            sort_lexicographical=_bool(
                first(model, "./*[local-name()='entry' and @key='sort_lexicographical']/@value"), False
            ),
        )

    @staticmethod
    def emit(settings: PivotSettings) -> List[str]:
        return [
            f"_group_cols = {repr(settings.group_cols)}",
            f"_pivot_cols = {repr(settings.pivot_cols)}",
            f"_aggregations = {repr([agg.__dict__ for agg in settings.aggregations])}",
            f"_missing_values = {repr(settings.missing_values)}",
            f"_sort = {repr(settings.sort_lexicographical)}",
            "_group_cols = [c for c in _group_cols if c in df.columns]",
            "_pivot_cols = [c for c in _pivot_cols if c in df.columns]",
            "_aggregations = [a for a in _aggregations if a.get('column') in df.columns]",
            "",
            "def _pivot_method_parts(method):",
            "    raw = str(method or 'Count').strip() or 'Count'",
            "    label = raw.split('_V', 1)[0]",
            "    key = label.strip().lower().replace(' ', '_')",
            "    mapping = {",
            "        'count': 'count',",
            "        'count_unique': 'nunique',",
            "        'unique_count': 'nunique',",
            "        'sum': 'sum',",
            "        'mean': 'mean',",
            "        'average': 'mean',",
            "        'median': 'median',",
            "        'minimum': 'min',",
            "        'min': 'min',",
            "        'maximum': 'max',",
            "        'max': 'max',",
            "        'standard_deviation': 'std',",
            "        'stddev': 'std',",
            "        'variance': 'var',",
            "        'var': 'var',",
            "        'first': 'first',",
            "        'last': 'last',",
            "        'concatenate': lambda s: ', '.join(s.dropna().astype(str)),",
            "    }",
            "    return label, mapping.get(key, key or 'count')",
            "",
            "def _pivot_flatten_columns(frame, value_cols, labels_by_col):",
            "    if not isinstance(frame.columns, pd.MultiIndex):",
            "        frame.columns = [str(c) for c in frame.columns]",
            "        return frame",
            "    flat_cols = []",
            "    single_value = value_cols[0] if len(value_cols) == 1 else None",
            "    for col_tuple in frame.columns.to_flat_index():",
            "        parts = tuple(col_tuple) if isinstance(col_tuple, tuple) else (col_tuple,)",
            "        if parts and parts[0] in value_cols:",
            "            value_col = parts[0]",
            "            pivot_values = parts[1:]",
            "        else:",
            "            value_col = single_value",
            "            pivot_values = parts",
            "        pivot_name = '_'.join('Missing' if pd.isna(p) else str(p) for p in pivot_values if str(p) != '')",
            "        label = labels_by_col.get(value_col, 'Count')",
            "        flat_cols.append(f'{pivot_name}+{label}({value_col})' if pivot_name else f'{label}({value_col})')",
            "    frame.columns = flat_cols",
            "    return frame",
            "",
            "def _pivot_make(index_cols):",
            "    value_cols = [a.get('column') for a in _aggregations]",
            "    labels_by_col = {}",
            "    aggfunc = {}",
            "    for a in _aggregations:",
            "        label, func = _pivot_method_parts(a.get('method'))",
            "        labels_by_col[a.get('column')] = label",
            "        aggfunc[a.get('column')] = func",
            "    if not _pivot_cols or not value_cols:",
            "        return pd.DataFrame(columns=list(index_cols))",
            "    source = df.copy()",
            "    effective_index = list(index_cols)",
            "    if not effective_index:",
            "        source['_k2p_total'] = 'Total'",
            "        effective_index = ['_k2p_total']",
            "    if _missing_values:",
            "        source[_pivot_cols] = source[_pivot_cols].astype('object').where(source[_pivot_cols].notna(), 'Missing')",
            "    table = pd.pivot_table(",
            "        source,",
            "        values=value_cols,",
            "        index=effective_index,",
            "        columns=_pivot_cols,",
            "        aggfunc=aggfunc,",
            "        dropna=not _missing_values,",
            "        sort=_sort,",
            "    )",
            "    if isinstance(table, pd.Series):",
            "        table = table.to_frame()",
            "    table = _pivot_flatten_columns(table, value_cols, labels_by_col).reset_index()",
            "    if '_k2p_total' in table.columns:",
            "        table = table.drop(columns=['_k2p_total'])",
            "    return table",
            "",
            "def _group_totals_make():",
            "    if not _group_cols or not _aggregations:",
            "        return pd.DataFrame()",
            "    named = {}",
            "    for a in _aggregations:",
            "        label, func = _pivot_method_parts(a.get('method'))",
            "        named[f'{label}({a.get(\"column\")})'] = (a.get('column'), func)",
            "    return df.groupby(_group_cols, dropna=False, sort=_sort).agg(**named).reset_index()",
            "",
            "out_df = _pivot_make(_group_cols) if _pivot_cols else df.copy()",
            "group_totals_df = _group_totals_make()",
            "overall_totals_df = _pivot_make([])",
        ]


def parse_pivot_settings(node_dir: Optional[Path]) -> PivotSettings:
    return Pivot.parse(node_dir)


def generate_imports() -> List[str]:
    return ["import pandas as pd"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_pivot_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']  # input table",
    ]
    lines.extend(Pivot.emit(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        if port == "2":
            lines.append(f"context['{node_id}:{port}'] = group_totals_df")
        elif port == "3":
            lines.append(f"context['{node_id}:{port}'] = overall_totals_df")
        else:
            lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "Pivot"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
