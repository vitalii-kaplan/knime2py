#!/usr/bin/env python3

"""
Missing Value (Apply) node handler.

This node consumes a Missing Value model (produced by the learner node) and applies
the stored strategies to a new dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .node_utils import collect_module_imports, normalize_in_ports, split_out_imports

FACTORY = "org.knime.base.node.preproc.pmml.missingval.apply.MissingValueApplyNodeFactory"
HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.pmml.missingval.apply.MissingValueApplyNodeFactory"
)


def generate_imports() -> List[str]:
    return [
        "import json",
        "import pandas as pd",
        "import xml.etree.ElementTree as ET",
    ]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    pairs = normalize_in_ports(in_ports)
    if len(pairs) < 2:
        raise ValueError("Missing Value (Apply) expects data and model inputs.")

    (data_src, data_port), (model_src, model_port) = pairs[0], pairs[1]

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"df = context['{data_src}:{data_port}']")
    lines.append(f"model = context.get('{model_src}:{model_port}')")
    lines.append("def _load_metadata(model_obj):")
    lines.append("    if isinstance(model_obj, dict):")
    lines.append("        type_strats = model_obj.get('strategies')")
    lines.append("        col_strats = model_obj.get('column_strategies', [])")
    lines.append("        if isinstance(type_strats, list):")
    lines.append("            return {")
    lines.append("                'type_strategies': type_strats,")
    lines.append("                'column_strategies': col_strats if isinstance(col_strats, list) else [],")
    lines.append("            }")
    lines.append("    text = None")
    lines.append("    if isinstance(model_obj, (bytes, bytearray)):")
    lines.append("        text = model_obj.decode('utf-8', errors='ignore')")
    lines.append("    elif isinstance(model_obj, str):")
    lines.append("        text = model_obj")
    lines.append("    if not text:")
    lines.append("        return {'type_strategies': [], 'column_strategies': []}")
    lines.append("    try:")
    lines.append("        root = ET.fromstring(text)")
    lines.append("    except ET.ParseError:")
    lines.append("        return {'type_strategies': [], 'column_strategies': []}")
    lines.append("    payload = None")
    lines.append("    for ext in root.findall('.//{*}Extension'):")
    lines.append("        if (ext.get('name') or '').strip() == 'missing_value_metadata':")
    lines.append("            payload = (ext.text or '').strip()")
    lines.append("            if payload:")
    lines.append("                break")
    lines.append("    if not payload:")
    lines.append("        return {'type_strategies': [], 'column_strategies': []}")
    lines.append("    try:")
    lines.append("        data = json.loads(payload)")
    lines.append("    except json.JSONDecodeError:")
    lines.append("        return {'type_strategies': [], 'column_strategies': []}")
    lines.append("    if not isinstance(data, dict):")
    lines.append("        return {'type_strategies': [], 'column_strategies': []}")
    lines.append("    type_strats = data.get('type_strategies')")
    lines.append("    col_strats = data.get('column_strategies')")
    lines.append("    return {")
    lines.append("        'type_strategies': type_strats if isinstance(type_strats, list) else [],")
    lines.append("        'column_strategies': col_strats if isinstance(col_strats, list) else [],")
    lines.append("    }")
    lines.append("meta = _load_metadata(model)")
    lines.append("type_strategies = meta.get('type_strategies', []) or []")
    lines.append("column_strategies = meta.get('column_strategies', []) or []")
    lines.append("out_df = df.copy()")
    lines.append("override_cols = set()")
    lines.append("for entry in column_strategies:")
    lines.append("    column = entry.get('column')")
    lines.append("    if not column or column not in out_df.columns:")
    lines.append("        continue")
    lines.append("    dtype = str(entry.get('dtype') or '').lower()")
    lines.append("    strategy = str(entry.get('strategy') or '').lower()")
    lines.append("    value = entry.get('value')")
    lines.append("    override_cols.add(str(column))")
    lines.append("    if strategy == 'fixed':")
    lines.append("        if value is None:")
    lines.append("            continue")
    lines.append("        if dtype == 'int':")
    lines.append("            try:")
    lines.append("                lit = int(value)")
    lines.append("            except Exception:")
    lines.append("                continue")
    lines.append("            out_df[column] = out_df[column].fillna(lit).astype('Int64')")
    lines.append("        elif dtype == 'float':")
    lines.append("            try:")
    lines.append("                lit = float(value)")
    lines.append("            except Exception:")
    lines.append("                continue")
    lines.append("            out_df[column] = out_df[column].fillna(lit)")
    lines.append("        elif dtype == 'boolean':")
    lines.append("            lit = str(value).strip().lower()")
    lines.append("            val = True if lit in {'true','1','t','y','yes'} else False")
    lines.append("            out_df[column] = out_df[column].fillna(val).astype('boolean')")
    lines.append("        else:")
    lines.append("            out_df[column] = out_df[column].fillna(str(value))")
    lines.append("    elif strategy in ('mean', 'median'):")
    lines.append("        fn = 'mean' if strategy == 'mean' else 'median'")
    lines.append("        stat = getattr(out_df[column], fn)()")
    lines.append("        if pd.isna(stat):")
    lines.append("            continue")
    lines.append("        filled = out_df[column].fillna(stat)")
    lines.append("        if dtype == 'int':")
    lines.append("            out_df[column] = filled.round().astype('Int64')")
    lines.append("        elif dtype == 'boolean':")
    lines.append("            out_df[column] = filled.astype('boolean')")
    lines.append("        else:")
    lines.append("            out_df[column] = filled")
    lines.append("    elif strategy == 'mode':")
    lines.append("        mode = out_df[column].mode()")
    lines.append("        if mode.empty:")
    lines.append("            continue")
    lines.append("        filled = out_df[column].fillna(mode.iloc[0])")
    lines.append("        if dtype == 'int':")
    lines.append("            out_df[column] = filled.astype('Int64')")
    lines.append("        elif dtype == 'boolean':")
    lines.append("            out_df[column] = filled.astype('boolean')")
    lines.append("        else:")
    lines.append("            out_df[column] = filled")
    lines.append("    elif strategy == 'ffill':")
    lines.append("        filled = out_df[column].ffill()")
    lines.append("        if dtype == 'int':")
    lines.append("            out_df[column] = filled.astype('Int64')")
    lines.append("        elif dtype == 'boolean':")
    lines.append("            out_df[column] = filled.astype('boolean')")
    lines.append("        else:")
    lines.append("            out_df[column] = filled")
    lines.append("    elif strategy == 'bfill':")
    lines.append("        filled = out_df[column].bfill()")
    lines.append("        if dtype == 'int':")
    lines.append("            out_df[column] = filled.astype('Int64')")
    lines.append("        elif dtype == 'boolean':")
    lines.append("            out_df[column] = filled.astype('boolean')")
    lines.append("        else:")
    lines.append("            out_df[column] = filled")
    lines.append("    elif strategy == 'drop':")
    lines.append("        out_df = out_df.dropna(subset=[column])")

    lines.append("for entry in type_strategies:")
    lines.append("    dtype = str(entry.get('dtype', '')).lower()")
    lines.append("    strategy = str(entry.get('strategy', '')).lower()")
    lines.append("    value = entry.get('value')")
    lines.append("    if dtype == 'int':")
    lines.append("        cols = [c for c in out_df.select_dtypes(include=['Int64','Int32','Int16','int64','int32','int16']).columns if c not in override_cols]")
    lines.append("        if len(cols) == 0:")
    lines.append("            continue")
    lines.append("        if strategy == 'fixed':")
    lines.append("            if value is None:")
    lines.append("                continue")
    lines.append("            try:")
    lines.append("                val = int(value)")
    lines.append("            except Exception:")
    lines.append("                continue")
    lines.append("            out_df[cols] = out_df[cols].fillna(val).astype('Int64')")
    lines.append("        elif strategy in ('mean','median'):")
    lines.append("            fn = 'mean' if strategy == 'mean' else 'median'")
    lines.append("            out_df[cols] = out_df[cols].apply(lambda s: (s if pd.isna(getattr(s, fn)()) else s.fillna(getattr(s, fn)()).round()).astype('Int64'))")
    lines.append("        elif strategy == 'mode':")
    lines.append("            out_df[cols] = out_df[cols].apply(lambda s: (s.fillna(s.mode().iloc[0]) if not s.mode().empty else s).astype('Int64'))")
    lines.append("        elif strategy == 'ffill':")
    lines.append("            out_df[cols] = out_df[cols].apply(lambda s: s.ffill().astype('Int64'))")
    lines.append("        elif strategy == 'bfill':")
    lines.append("            out_df[cols] = out_df[cols].apply(lambda s: s.bfill().astype('Int64'))")
    lines.append("        elif strategy == 'drop':")
    lines.append("            out_df = out_df.dropna(subset=cols.tolist())")
    lines.append("    elif dtype == 'float':")
    lines.append("        cols = [c for c in out_df.select_dtypes(include=['float64','float32']).columns if c not in override_cols]")
    lines.append("        if len(cols) == 0:")
    lines.append("            continue")
    lines.append("        if strategy == 'fixed':")
    lines.append("            if value is None:")
    lines.append("                continue")
    lines.append("            try:")
    lines.append("                val = float(value)")
    lines.append("            except Exception:")
    lines.append("                continue")
    lines.append("            out_df[cols] = out_df[cols].fillna(val)")
    lines.append("        elif strategy in ('mean','median'):")
    lines.append("            fn = 'mean' if strategy == 'mean' else 'median'")
    lines.append("            out_df[cols] = out_df[cols].apply(lambda s: s.fillna(getattr(s, fn)()))")
    lines.append("        elif strategy == 'mode':")
    lines.append("            out_df[cols] = out_df[cols].apply(lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else s))")
    lines.append("        elif strategy == 'ffill':")
    lines.append("            out_df[cols] = out_df[cols].ffill()")
    lines.append("        elif strategy == 'bfill':")
    lines.append("            out_df[cols] = out_df[cols].bfill()")
    lines.append("        elif strategy == 'drop':")
    lines.append("            out_df = out_df.dropna(subset=cols.tolist())")
    lines.append("    elif dtype == 'string':")
    lines.append("        cols = [c for c in out_df.select_dtypes(include=['string','object']).columns if c not in override_cols]")
    lines.append("        if len(cols) == 0:")
    lines.append("            continue")
    lines.append("        if strategy == 'fixed':")
    lines.append("            if value is None:")
    lines.append("                continue")
    lines.append("            out_df[cols] = out_df[cols].fillna(str(value))")
    lines.append("        elif strategy == 'mode':")
    lines.append("            out_df[cols] = out_df[cols].apply(lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else s))")
    lines.append("        elif strategy == 'ffill':")
    lines.append("            out_df[cols] = out_df[cols].ffill()")
    lines.append("        elif strategy == 'bfill':")
    lines.append("            out_df[cols] = out_df[cols].bfill()")
    lines.append("        elif strategy == 'drop':")
    lines.append("            out_df = out_df.dropna(subset=cols.tolist())")
    lines.append("    elif dtype == 'boolean':")
    lines.append("        cols = [c for c in out_df.select_dtypes(include=['boolean','bool']).columns if c not in override_cols]")
    lines.append("        if len(cols) == 0:")
    lines.append("            continue")
    lines.append("        if strategy == 'fixed':")
    lines.append("            if value is None:")
    lines.append("                continue")
    lines.append("            lit = str(value).strip().lower()")
    lines.append("            val = True if lit in {'true','1','t','y','yes'} else False")
    lines.append("            out_df[cols] = out_df[cols].fillna(val).astype('boolean')")
    lines.append("        elif strategy == 'mode':")
    lines.append("            out_df[cols] = out_df[cols].apply(lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else s)).astype('boolean')")
    lines.append("        elif strategy == 'ffill':")
    lines.append("            out_df[cols] = out_df[cols].ffill().astype('boolean')")
    lines.append("        elif strategy == 'bfill':")
    lines.append("            out_df[cols] = out_df[cols].bfill().astype('boolean')")
    lines.append("        elif strategy == 'drop':")
    lines.append("            out_df = out_df.dropna(subset=cols.tolist())")

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
