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
    return ["import pandas as pd"]


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
    lines.append("strategies = []")
    lines.append("if isinstance(model, dict):")
    lines.append("    raw = model.get('strategies')")
    lines.append("    if isinstance(raw, list):")
    lines.append("        strategies = raw")
    lines.append("out_df = df.copy()")
    lines.append("for entry in strategies:")
    lines.append("    dtype = str(entry.get('dtype', '')).lower()")
    lines.append("    strategy = str(entry.get('strategy', '')).lower()")
    lines.append("    value = entry.get('value')")
    lines.append("    if dtype == 'int':")
    lines.append("        cols = out_df.select_dtypes(include=['Int64','Int32','Int16','int64','int32','int16']).columns")
    lines.append("        if len(cols) == 0:")
    lines.append("            continue")
    lines.append("        if strategy == 'fixed':")
    lines.append("            if value is None:")
    lines.append("                continue")
    lines.append("            lit = value if isinstance(value, (int, float)) else str(value)")
    lines.append("            try:")
    lines.append("                val = int(lit)")
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
    lines.append("        cols = out_df.select_dtypes(include=['float64','float32']).columns")
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
    lines.append("        cols = out_df.select_dtypes(include=['string','object']).columns")
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
    lines.append("        cols = out_df.select_dtypes(include=['boolean','bool']).columns")
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
