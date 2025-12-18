#!/usr/bin/env python3

"""
Normalizer (PMML) node handler.

This node mirrors the behavior of the standard Normalizer but emits a PMML document
describing the applied transformations on its second port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

FACTORY = "org.knime.base.node.preproc.pmml.normalize.NormalizerPMMLNodeFactory2"


@dataclass
class NormalizerPMMLSettings:
    mode: str = "MINMAX"  # MINMAX or ZSCORE
    new_min: float = 0.0
    new_max: float = 1.0
    columns: List[str] = field(default_factory=list)
    all_numeric: bool = False


def _as_bool(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() == "true"


def parse_settings(node_dir: Optional[Path]) -> NormalizerPMMLSettings:
    if not node_dir:
        return NormalizerPMMLSettings()
    sp = node_dir / "settings.xml"
    if not sp.exists():
        return NormalizerPMMLSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return NormalizerPMMLSettings()

    raw_mode = first(model, ".//*[local-name()='entry' and @key='mode']/@value") or "2"
    mode = "MINMAX"
    if raw_mode.strip() in {"0", "1"}:
        mode = "ZSCORE"

    new_min = first(model, ".//*[local-name()='entry' and translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='newmin']/@value")
    new_max = first(model, ".//*[local-name()='entry' and translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='newmax']/@value")

    try:
        min_val = float(new_min) if new_min is not None else 0.0
    except Exception:
        min_val = 0.0
    try:
        max_val = float(new_max) if new_max is not None else 1.0
    except Exception:
        max_val = 1.0

    cols_cfg = first_el(model, ".//*[local-name()='config' and @key='columns']")
    cols: List[str] = []
    if cols_cfg is not None:
        for entry in cols_cfg.xpath("./*[local-name()='entry']"):
            key = (entry.get("key") or "").strip()
            val = entry.get("value") or ""
            if key.isdigit():
                cols.append(val)

    all_numeric = _as_bool(first(model, ".//*[local-name()='entry' and @key='all_numeric_columns_used']/@value"))

    return NormalizerPMMLSettings(
        mode=mode,
        new_min=min_val,
        new_max=max_val,
        columns=cols,
        all_numeric=all_numeric,
    )


def generate_imports() -> List[str]:
    return ["import pandas as pd", "from html import escape"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.pmml.normalize.NormalizerPMMLNodeFactory2"
)


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    src_id, in_port = normalize_in_ports(in_ports)[0]
    lines.append(f"df = context['{src_id}:{in_port}']")
    lines.append("out_df = df.copy()")

    if cfg.columns:
        lines.append(f"cand_cols = {repr(cfg.columns)}")
    elif cfg.all_numeric:
        lines.append("cand_cols = out_df.select_dtypes(include=['number']).columns.tolist()")
    else:
        lines.append("cand_cols = out_df.columns.tolist()")

    lines.append("norm_cols = [c for c in cand_cols if c in out_df.columns]")
    lines.append("pmml_stats = {}")
    lines.append("if not norm_cols:")
    lines.append("    pass")
    lines.append("else:")
    lines.append("    out_df[norm_cols] = out_df[norm_cols].apply(pd.to_numeric, errors='coerce')")
    lines.append("    pmml_stats = {}")

    if cfg.mode == "MINMAX":
        lines.append("    _col_min = out_df[norm_cols].min(axis=0, skipna=True)")
        lines.append("    _col_max = out_df[norm_cols].max(axis=0, skipna=True)")
        lines.append("    def _minmax_col(series):")
        lines.append("        mn = _col_min.get(series.name)")
        lines.append("        mx = _col_max.get(series.name)")
        lines.append("        if mn is None or mx is None or pd.isna(mn) or pd.isna(mx) or mx == mn:")
        lines.append(f"            pmml_stats[series.name] = {{'min': None, 'max': None}}")
        lines.append(f"            return pd.Series([{cfg.new_min}] * len(series), index=series.index)")
        lines.append("        pmml_stats[series.name] = {'min': float(mn), 'max': float(mx)}")
        lines.append(f"        return ({cfg.new_min} + (series - mn) / (mx - mn) * ({cfg.new_max - cfg.new_min})).astype(float)")
        lines.append("    out_df[norm_cols] = out_df[norm_cols].apply(_minmax_col)")
    else:
        lines.append("    _col_mean = out_df[norm_cols].mean(axis=0, skipna=True)")
        lines.append("    _col_std = out_df[norm_cols].std(axis=0, ddof=0, skipna=True)")
        lines.append("    def _zscore_col(series):")
        lines.append("        mu = _col_mean.get(series.name)")
        lines.append("        sd = _col_std.get(series.name)")
        lines.append("        if sd is None or pd.isna(sd) or sd == 0:")
        lines.append("            pmml_stats[series.name] = {'mean': float(mu) if mu is not None else 0.0, 'std': None}")
        lines.append("            return pd.Series([0.0] * len(series), index=series.index)")
        lines.append("        pmml_stats[series.name] = {'mean': float(mu) if mu is not None else 0.0, 'std': float(sd)}")
        lines.append("        center = 0.0 if mu is None or pd.isna(mu) else mu")
        lines.append("        return ((series - center) / sd).astype(float)")
        lines.append("    out_df[norm_cols] = out_df[norm_cols].apply(_zscore_col)")

    lines.append("pmml_lines = [")
    lines.append("    \"<?xml version='1.0' encoding='UTF-8'?>\",")
    lines.append("    \"<PMML version='4.4' xmlns='http://www.dmg.org/PMML-4_4'>\",")
    lines.append("    \"  <Header>\",")
    lines.append("    \"    <Application name='knime2py' version='1.0'/>\",")
    lines.append("    \"  </Header>\",")
    lines.append("]")
    lines.append("if norm_cols:")
    lines.append("    pmml_lines.append(\"  <TransformationDictionary>\")")
    lines.append("    for col in norm_cols:")
    lines.append("        escaped = escape(str(col))")
    if cfg.mode == "MINMAX":
        lines.append("        params = pmml_stats.get(col) or {}")
        lines.append("        mn = params.get('min')")
        lines.append("        mx = params.get('max')")
        lines.append("        if mn is None or mx is None:")
        lines.append("            continue")
        lines.append("        pmml_lines.append(f\"    <DerivedField name='{escaped}_norm' optype='continuous' dataType='double'>\")")
        lines.append("        pmml_lines.append(f\"      <NormContinuous origField='{escaped}'>\")")
        lines.append(f"        pmml_lines.append(f\"        <LinearNorm orig='{{mn}}' norm='{cfg.new_min}'/>\")")
        lines.append(f"        pmml_lines.append(f\"        <LinearNorm orig='{{mx}}' norm='{cfg.new_max}'/>\")")
        lines.append("        pmml_lines.append(\"      </NormContinuous>\")")
        lines.append("        pmml_lines.append(\"    </DerivedField>\")")
    else:
        lines.append("        params = pmml_stats.get(col) or {}")
        lines.append("        mu = params.get('mean')")
        lines.append("        sd = params.get('std')")
        lines.append("        if sd is None or sd == 0:")
        lines.append("            continue")
        lines.append("        pmml_lines.append(f\"    <DerivedField name='{escaped}_z' optype='continuous' dataType='double'>\")")
        lines.append("        pmml_lines.append(\"      <Apply function='/'><Apply function='-'>\")")
        lines.append("        pmml_lines.append(f\"        <FieldRef field='{escaped}'/>\")")
        lines.append("        pmml_lines.append(f\"        <Constant dataType='double'>{{mu}}</Constant></Apply>\")")
        lines.append("        pmml_lines.append(f\"        <Constant dataType='double'>{{sd}}</Constant>\")")
        lines.append("        pmml_lines.append(\"      </Apply>\")")
        lines.append("        pmml_lines.append(\"    </DerivedField>\")")
    lines.append("    pmml_lines.append(\"  </TransformationDictionary>\")")
    lines.append("pmml_lines.append(\"</PMML>\")")
    lines.append("pmml_model = \"\\n\".join(pmml_lines)")

    ports = out_ports or ["1", "2"]
    port_map = {"1": "out_df", "2": "pmml_model"}
    for p in sorted({(p or '1') for p in ports}):
        target = port_map.get(p, "out_df")
        lines.append(f"context['{node_id}:{p}'] = {target}")

    return lines


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src, str(getattr(edge, "source_port", "") or "1")) for src, edge in (incoming or [])]
    out_ports = [str(getattr(edge, "source_port", "") or "1") for _, edge in (outgoing or [])] or ["1", "2"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
