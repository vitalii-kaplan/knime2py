#!/usr/bin/env python3

"""
String Manipulation node handler.

This module implements a limited subset of the KNIME String Manipulation node,
mirroring the Math Formula handler but targeting pandas string operations.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import (
    collect_module_imports,
    first,
    first_el,
    normalize_in_ports,
    split_out_imports,
)

FACTORY = "org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory"
HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory"
)

_COL_TOKEN = re.compile(r"\$(.+?)\$")


@dataclass
class StringManipSettings:
    expression: str = ""
    append: bool = False
    new_col_name: str = "String Manipulation"
    replace_col: Optional[str] = None
    abort_on_error: bool = True
    insert_missing: bool = True


def _translate_expression(expr: str) -> str:
    """Translate KNIME expression to Python code."""
    s = html.unescape(expr or "").strip()
    if not s:
        return ""

    def repl_col(match: re.Match) -> str:
        col = match.group(1)
        return f"df[{repr(col)}]"

    s = _COL_TOKEN.sub(repl_col, s)
    replacements = {
        "substr(": "_substr(",
        "lowerCase(": "_lower(",
        "upperCase(": "_upper(",
        "length(": "_length(",
        "trim(": "_trim(",
        "replace(": "_replace(",
        "regexReplace(": "_regex_replace(",
        "indexOf(": "_index_of(",
        "count(": "_count_occurrences(",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s


def parse_string_manip_settings(node_dir: Optional[Path]) -> StringManipSettings:
    if not node_dir:
        return StringManipSettings()
    sp = node_dir / "settings.xml"
    if not sp.exists():
        return StringManipSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")

    if model is None:
        return StringManipSettings()

    expr = first(model, ".//*[local-name()='entry' and @key='expression']/@value") or ""
    expr = expr.strip()
    append = first(model, ".//*[local-name()='entry' and @key='append_column']/@value")
    append_flag = (append or "").strip().lower() == "true"
    new_col = first(model, ".//*[local-name()='entry' and @key='appended_column_name']/@value")
    replace_col = first(model, ".//*[local-name()='entry' and @key='replaced_column']/@value")
    abort = first(model, ".//*[local-name()='entry' and @key='abort_execution_on_evaluation_errors']/@value")
    insert = first(model, ".//*[local-name()='entry' and @key='insert_missing_as_null']/@value")

    return StringManipSettings(
        expression=expr,
        append=append_flag,
        new_col_name=(new_col or "String Manipulation"),
        replace_col=(replace_col or None),
        abort_on_error=(abort or "true").strip().lower() == "true",
        insert_missing=(insert or "true").strip().lower() == "true",
    )


def generate_imports() -> List[str]:
    return ["import pandas as pd", "import re"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_string_manip_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    src_id, in_port = normalize_in_ports(in_ports)[0]
    lines.append(f"df = context['{src_id}:{in_port}']")
    lines.append("out_df = df.copy()")
    lines += [
        "def _as_series(obj):",
        "    if isinstance(obj, pd.Series):",
        "        return obj.reindex(df.index)",
        "    try:",
        "        return pd.Series(obj, index=df.index)",
        "    except Exception:",
        "        return pd.Series([obj] * len(df.index), index=df.index)",
        "",
        "def _broadcast(value, index):",
        "    if isinstance(value, pd.Series):",
        "        return value.reindex(index)",
        "    return pd.Series(value, index=index)",
        "",
        "def _substr(series, start, length=None):",
        "    s = _as_series(series).astype('string')",
        "    start_series = _broadcast(start, s.index)",
        "    length_series = pd.Series([None] * len(s), index=s.index) if length is None else _broadcast(length, s.index)",
        "",
        "    def _slice_one(val, start_val, len_val):",
        "        if pd.isna(val):",
        "            return pd.NA",
        "        start_idx = 0 if pd.isna(start_val) else int(start_val)",
        "        if pd.isna(len_val):",
        "            return str(val)[start_idx:]",
        "        stop_idx = start_idx + int(len_val)",
        "        return str(val)[start_idx:stop_idx]",
        "",
        "    return pd.Series((",
        "        _slice_one(val, st, ln)",
        "        for val, st, ln in zip(s, start_series, length_series)",
        "    ), index=s.index, dtype='string')",
        "",
        "def _lower(series):",
        "    return _as_series(series).astype('string').str.lower()",
        "",
        "def _upper(series):",
        "    return _as_series(series).astype('string').str.upper()",
        "",
        "def _length(series):",
        "    return _as_series(series).astype('string').str.len()",
        "",
        "def _trim(series):",
        "    return _as_series(series).astype('string').str.strip()",
        "",
        "def _replace(series, old, new):",
        "    return _as_series(series).astype('string').str.replace(str(old), str(new), regex=False)",
        "",
        "def _regex_replace(series, pattern, repl, flags=0):",
        "    return _as_series(series).astype('string').str.replace(str(pattern), str(repl), regex=True)",
        "",
        "def _index_of(series, needle, occurrence=None):",
        "    s = _as_series(series).astype('string')",
        "    needle_series = _broadcast(needle, s.index)",
        "    occ_series = pd.Series([None] * len(s), index=s.index) if occurrence is None else _broadcast(occurrence, s.index)",
        "",
        "    def _nth(val, target, occ_val):",
        "        if pd.isna(val):",
        "            return pd.NA",
        "        text = str(val)",
        "        tgt = '' if pd.isna(target) else str(target)",
        "        occ = 1",
        "        if not pd.isna(occ_val):",
        "            try:",
        "                occ = max(1, int(occ_val))",
        "            except Exception:",
        "                occ = 1",
        "        idx = -1",
        "        start = 0",
        "        remaining = occ",
        "        while remaining > 0:",
        "            idx = text.find(tgt, start)",
        "            if idx == -1:",
        "                break",
        "            start = idx + 1",
        "            remaining -= 1",
        "        return idx",
        "",
        "    return pd.Series((",
        "        _nth(val, tgt, occ_val)",
        "        for val, tgt, occ_val in zip(s, needle_series, occ_series)",
        "    ), index=s.index)",
        "",
        "def _count_occurrences(series, needle):",
        "    pat = re.escape(str(needle))",
        "    return _as_series(series).astype('string').str.count(pat)",
        "",
    ]

    expr_code = _translate_expression(cfg.expression)
    if not expr_code:
        lines.append("result_series = pd.Series(pd.NA, index=df.index)")
    else:
        lines.append("_abort_on_error = " + ("True" if cfg.abort_on_error else "False"))
        lines.append("_insert_missing = " + ("True" if cfg.insert_missing else "False"))
        lines.append("try:")
        lines.append(f"    result_series = {expr_code}")
        lines.append("except Exception as exc:")
        lines.append("    if _abort_on_error:")
        lines.append("        raise")
        lines.append("    if _insert_missing:")
        lines.append("        result_series = pd.Series(pd.NA, index=df.index)")
        lines.append("    else:")
        lines.append("        result_series = pd.Series('', index=df.index)")

    lines.append("result_series = _as_series(result_series)")

    target_col = cfg.new_col_name if cfg.append else (cfg.replace_col or cfg.new_col_name)
    lines.append(f"out_df[{repr(target_col)}] = result_series")

    ports = out_ports or ["1"]
    for p in sorted({(p or '1') for p in ports}):
        lines.append(f"context['{node_id}:{p}'] = out_df")
    return lines


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src, str(getattr(e, "source_port", "") or "1")) for src, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
