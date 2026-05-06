#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, iter_entries, normalize_in_ports, split_out_imports


FACTORY = "org.knime.base.node.preproc.groupby.GroupByNodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.groupby.GroupByNodeFactory"
)


@dataclass
class Aggregation:
    column: str
    method: str
    include_missing: bool = False


@dataclass
class GroupBySettings:
    group_cols: List[str] = field(default_factory=list)
    aggregations: List[Aggregation] = field(default_factory=list)
    column_name_policy: str = "Aggregation method (column name)"
    retain_order: bool = False


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


def parse_groupby_settings(node_dir: Optional[Path]) -> GroupBySettings:
    if not node_dir:
        return GroupBySettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return GroupBySettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return GroupBySettings()

    group_cols = _indexed_values(
        first_el(
            model,
            "./*[local-name()='config' and @key='grouByColumns']"
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

    aggregations: List[Aggregation] = []
    for idx, column in enumerate(columns):
        if not column:
            continue
        method = methods[idx] if idx < len(methods) and methods[idx] else "Count"
        include_missing = _bool(missing[idx], False) if idx < len(missing) else False
        aggregations.append(Aggregation(column=column, method=method, include_missing=include_missing))

    return GroupBySettings(
        group_cols=group_cols,
        aggregations=aggregations,
        column_name_policy=first(model, "./*[local-name()='entry' and @key='columnNamePolicy']/@value")
        or "Aggregation method (column name)",
        retain_order=_bool(first(model, "./*[local-name()='entry' and @key='retainOrder']/@value"), False),
    )


def generate_imports() -> List[str]:
    return ["import pandas as pd"]


def _emit_groupby_code(settings: GroupBySettings) -> List[str]:
    lines: List[str] = [
        f"_group_cols = {repr(settings.group_cols)}",
        f"_aggregations = {repr([agg.__dict__ for agg in settings.aggregations])}",
        f"_retain_order = {repr(settings.retain_order)}",
        "_group_cols = [c for c in _group_cols if c in df.columns]",
        "",
        "def _gb_method_key(method):",
        "    raw = str(method or '').strip()",
        "    key = raw.lower().replace(' ', '_')",
        "    if '_v' in key:",
        "        key = key.split('_v', 1)[0]",
        "    return key",
        "",
        "def _gb_method_name(method):",
        "    key = _gb_method_key(method)",
        "    return {",
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
        "    }.get(key, key or 'count')",
        "",
        "def _gb_output_name(method, column):",
        "    key = _gb_method_key(method)",
        "    method_label = {",
        "        'count': 'Count',",
        "        'count_unique': 'Count unique',",
        "        'unique_count': 'Count unique',",
        "        'sum': 'Sum',",
        "        'mean': 'Mean',",
        "        'average': 'Mean',",
        "        'median': 'Median',",
        "        'minimum': 'Minimum',",
        "        'min': 'Minimum',",
        "        'maximum': 'Maximum',",
        "        'max': 'Maximum',",
        "        'standard_deviation': 'Standard deviation',",
        "        'stddev': 'Standard deviation',",
        "        'variance': 'Variance',",
        "        'var': 'Variance',",
        "        'first': 'First',",
        "        'last': 'Last',",
        "    }.get(key)",
        "    if not method_label:",
        "        method_label = str(method or 'Count').strip() or 'Count'",
        "    return f'{method_label}({column})'",
        "",
        "if not _group_cols:",
        "    out_df = df.copy()",
        "elif not _aggregations:",
        "    out_df = (",
        "        df.groupby(_group_cols, dropna=False, sort=not _retain_order)",
        "        .size()",
        "        .reset_index()",
        "        [_group_cols]",
        "    )",
        "else:",
        "    _named_aggs = {}",
        "    for _agg in _aggregations:",
        "        _col = _agg.get('column')",
        "        if _col not in df.columns:",
        "            continue",
        "        _method = _agg.get('method') or 'Count'",
        "        _func = _gb_method_name(_method)",
        "        _out = _gb_output_name(_method, _col)",
        "        if _out in _named_aggs:",
        "            _suffix = 2",
        "            _base = _out",
        "            while _out in _named_aggs:",
        "                _out = f'{_base} #{_suffix}'",
        "                _suffix += 1",
        "        _named_aggs[_out] = (_col, _func)",
        "    if _named_aggs:",
        "        out_df = df.groupby(_group_cols, dropna=False, sort=not _retain_order).agg(**_named_aggs).reset_index()",
        "    else:",
        "        out_df = df.groupby(_group_cols, dropna=False, sort=not _retain_order).size().reset_index()[_group_cols]",
    ]
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_groupby_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']  # input table",
    ]
    lines.extend(_emit_groupby_code(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "GroupBy"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
