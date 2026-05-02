#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import (
    collect_module_imports,
    first,
    first_el,
    normalize_delim,
    normalize_in_ports,
    split_out_imports,
)


FACTORY = "org.knime.base.node.preproc.colcombine2.ColCombine2NodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.colcombine2.ColCombine2NodeFactory"
)


@dataclass
class ColumnCombinerSettings:
    columns: list[str] = field(default_factory=list)
    delimiter: str = ","
    delimiter_inputs: str = "QUOTE"
    quote_char: str = '"'
    quote_inputs: str = "ONLY_NECESSARY"
    replace_delimiter: str = ""
    new_column_name: str = "Combined String"
    remove_included_columns: bool = False
    fail_if_missing_columns: bool = False


def _bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _parse_manual_selected(model: ET._Element) -> list[str]:
    selected = first_el(
        model,
        ".//*[local-name()='config' and @key='manualFilter']"
        "/*[local-name()='config' and @key='manuallySelected']",
    )
    if selected is None:
        return []

    numbered: list[tuple[int, str]] = []
    for entry in selected.xpath("./*[local-name()='entry']"):
        key = entry.get("key") or ""
        value = entry.get("value")
        if key.isdigit() and value:
            numbered.append((int(key), value))
    return [value for _, value in sorted(numbered, key=lambda item: item[0])]


def parse_column_combiner_settings(node_dir: Optional[Path]) -> ColumnCombinerSettings:
    if not node_dir:
        return ColumnCombinerSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return ColumnCombinerSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return ColumnCombinerSettings()

    delimiter = normalize_delim(first(model, "./*[local-name()='entry' and @key='delimiter']/@value")) or ","
    quote_char = first(model, "./*[local-name()='entry' and @key='quote_char']/@value") or '"'
    if quote_char == "&quot;":
        quote_char = '"'

    return ColumnCombinerSettings(
        columns=_parse_manual_selected(model),
        delimiter=delimiter,
        delimiter_inputs=(first(model, "./*[local-name()='entry' and @key='delimiterInputs']/@value") or "QUOTE").upper(),
        quote_char=quote_char[:1] or '"',
        quote_inputs=(first(model, "./*[local-name()='entry' and @key='quoteInputs']/@value") or "ONLY_NECESSARY").upper(),
        replace_delimiter=first(model, "./*[local-name()='entry' and @key='replace_delimiter']/@value") or "",
        new_column_name=first(model, "./*[local-name()='entry' and @key='new_column_name']/@value") or "Combined String",
        remove_included_columns=_bool(
            first(model, "./*[local-name()='entry' and @key='remove_included_columns']/@value"),
            False,
        ),
        fail_if_missing_columns=_bool(
            first(model, "./*[local-name()='entry' and @key='failIfMissingColumns']/@value"),
            False,
        ),
    )


def generate_imports() -> list[str]:
    return ["import pandas as pd"]


def _emit_combiner_code(settings: ColumnCombinerSettings) -> list[str]:
    return [
        "out_df = df.copy()",
        f"_combine_columns = {settings.columns!r}",
        f"_delimiter = {settings.delimiter!r}",
        f"_delimiter_inputs = {settings.delimiter_inputs!r}",
        f"_quote_char = {settings.quote_char!r}",
        f"_quote_inputs = {settings.quote_inputs!r}",
        f"_replace_delimiter = {settings.replace_delimiter!r}",
        f"_new_column_name = {settings.new_column_name!r}",
        f"_fail_if_missing_columns = {settings.fail_if_missing_columns!r}",
        "if _fail_if_missing_columns:",
        "    _missing_columns = [c for c in _combine_columns if c not in out_df.columns]",
        "    if _missing_columns:",
        "        raise KeyError(f\"Column Combiner missing columns: {_missing_columns}\")",
        "_combine_columns = [c for c in _combine_columns if c in out_df.columns]",
        "",
        "def _knime_combiner_cell(value):",
        "    if pd.isna(value):",
        "        return ''",
        "    text = str(value)",
        "    if _delimiter_inputs == 'REPLACE' and _delimiter:",
        "        text = text.replace(_delimiter, _replace_delimiter)",
        "    needs_quote = _quote_inputs == 'ALWAYS' or (",
        "        _quote_inputs == 'ONLY_NECESSARY'",
        "        and _quote_char",
        "        and (_delimiter in text or _quote_char in text or '\\n' in text or '\\r' in text)",
        "    )",
        "    if needs_quote:",
        "        escaped = text.replace(_quote_char, _quote_char + _quote_char)",
        "        return f'{_quote_char}{escaped}{_quote_char}'",
        "    return text",
        "",
        "out_df[_new_column_name] = out_df.apply(",
        "    lambda row: _delimiter.join(_knime_combiner_cell(row[col]) for col in _combine_columns),",
        "    axis=1,",
        ")",
        f"_remove_included_columns = {settings.remove_included_columns!r}",
        "if _remove_included_columns and _combine_columns:",
        "    out_df = out_df.drop(columns=_combine_columns, errors='ignore')",
    ]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: list[tuple[str, str]],
    out_ports: Optional[list[str]] = None,
) -> list[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_column_combiner_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']",
    ]
    lines.extend(_emit_combiner_code(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "Column Combiner"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
