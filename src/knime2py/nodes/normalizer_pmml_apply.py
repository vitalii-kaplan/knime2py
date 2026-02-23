#!/usr/bin/env python3

"""
Normalizer (PMML Apply) node handler.

Consumes a PMML-style normalization bundle (as produced by our Normalizer PMML node)
and applies the stored parameters to incoming data.
"""

from __future__ import annotations

from typing import List, Optional

from .node_utils import collect_module_imports, normalize_in_ports, split_out_imports

FACTORY = "org.knime.base.node.preproc.pmml.normalize.NormalizerPMMLApplyNodeFactory"
HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.pmml.normalize.NormalizerPMMLApplyNodeFactory"
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
        raise ValueError("Normalizer Apply (PMML) expects data and PMML inputs.")

    (data_src, data_port), (model_src, model_port) = pairs[0], pairs[1]

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"df = context['{data_src}:{data_port}']")
    lines.append(f"model = context.get('{model_src}:{model_port}')")
    lines.append("out_df = df.copy()")
    lines.append("bundle = model if isinstance(model, dict) else {}")
    lines.append("stats = bundle.get('stats')")
    lines.append("if not isinstance(stats, dict):")
    lines.append("    stats = {}")
    lines.append("mode = str(bundle.get('mode', 'MINMAX')).upper()")
    lines.append("columns = bundle.get('columns')")
    lines.append("if not isinstance(columns, list):")
    lines.append("    columns = list(stats.keys())")
    lines.append("norm_cols = [c for c in columns if c in out_df.columns]")
    lines.append("if not norm_cols or not stats:")
    lines.append("    pass  # nothing to normalize")
    lines.append("else:")
    lines.append("    out_df[norm_cols] = out_df[norm_cols].apply(pd.to_numeric, errors='coerce')")
    lines.append("    if mode == 'MINMAX':")
    lines.append("        new_min = float(bundle.get('new_min', 0.0))")
    lines.append("        new_max = float(bundle.get('new_max', 1.0))")
    lines.append("        span = new_max - new_min")
    lines.append("        for col in norm_cols:")
    lines.append("            params = stats.get(col) or {}")
    lines.append("            mn = params.get('min')")
    lines.append("            mx = params.get('max')")
    lines.append("            mn = None if mn is None or pd.isna(mn) else float(mn)")
    lines.append("            mx = None if mx is None or pd.isna(mx) else float(mx)")
    lines.append("            if mn is None or mx is None or pd.isna(mn) or pd.isna(mx) or mx == mn:")
    lines.append("                out_df[col] = pd.Series([new_min] * len(out_df), index=out_df.index)")
    lines.append("            else:")
    lines.append("                out_df[col] = (new_min + (out_df[col] - mn) / (mx - mn) * span).astype(float)")
    lines.append("    elif mode == 'ZSCORE':")
    lines.append("        for col in norm_cols:")
    lines.append("            params = stats.get(col) or {}")
    lines.append("            mu = params.get('mean')")
    lines.append("            sd = params.get('std')")
    lines.append("            mu = None if mu is None or pd.isna(mu) else float(mu)")
    lines.append("            sd = None if sd is None or pd.isna(sd) else float(sd)")
    lines.append("            if sd is None or pd.isna(sd) or sd == 0:")
    lines.append("                out_df[col] = pd.Series([0.0] * len(out_df), index=out_df.index)")
    lines.append("            else:")
    lines.append("                center = 0.0 if mu is None else mu")
    lines.append("                out_df[col] = ((out_df[col] - center) / sd).astype(float)")
    lines.append("    elif mode == 'DECIMALSCALING':")
    lines.append("        for col in norm_cols:")
    lines.append("            params = stats.get(col) or {}")
    lines.append("            scale = params.get('scale')")
    lines.append("            try:")
    lines.append("                scale_int = int(scale)")
    lines.append("            except Exception:")
    lines.append("                scale_int = None")
    lines.append("            if not scale_int:")
    lines.append("                out_df[col] = pd.Series([0.0] * len(out_df), index=out_df.index)")
    lines.append("            else:")
    lines.append("                denom = float(10 ** scale_int)")
    lines.append("                out_df[col] = (out_df[col] / denom).astype(float)")
    lines.append("    else:")
    lines.append("        pass")

    ports = out_ports or ["1"]
    for p in sorted({(p or '1') for p in ports}):
        lines.append(f"context['{node_id}:{p}'] = out_df")
    return lines



def get_name() -> str:
    """Return human-readable handler name."""
    return "Normalizer PMML Apply"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src, str(getattr(e, "source_port", "") or "1")) for src, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
