#!/usr/bin/env python3

####################################################################################################
#
# Row Aggregator
#
# Aggregates rows using a single aggregation function, optionally grouped by a category column.
# Supported methods: COUNT (occurrence count), SUM, AVERAGE, MINIMUM, MAXIMUM. For SUM/AVERAGE a
# weight column is honored when configured. The “aggregation columns” list is used for all methods
# except COUNT (COUNT simply counts rows per group).
# - Inputs: single table
# - Output 1: aggregated table (grouped by category if provided, otherwise a single-row table)
# - Output 2: optional “grand totals” (only if a category is configured and grandTotals=true)
#
# Keys used (model):
#   categoryColumn (string | null)
#   aggregationMethod (COUNT | SUM | AVERAGE | MINIMUM | MAXIMUM)
#   frequencyColumns/selected_Internals + manualFilter/manuallySelected  → aggregation column names
#   weightColumn (string | null)  — only SUM/AVERAGE use it
#   grandTotals (boolean)
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (  # helpers
    first,
    first_el,
    iter_entries,
    normalize_in_ports,
    collect_module_imports,
    split_out_imports,
)

# KNIME factory (Row Aggregator)
FACTORY = "org.knime.base.node.preproc.rowagg.RowAggregatorNodeFactory"

# ---------------------------------------------------------------------
# settings.xml → RowAggregatorSettings
# ---------------------------------------------------------------------

@dataclass
class RowAggregatorSettings:
    category_col: Optional[str] = None             # group-by column (optional)
    method: str = "COUNT"                          # COUNT | SUM | AVERAGE | MINIMUM | MAXIMUM
    agg_cols: List[str] = field(default_factory=list)
    weight_col: Optional[str] = None               # only for SUM/AVERAGE
    grand_totals: bool = False                     # second output without grouping (if category given)


def _collect_numeric_name_entries(cfg: Optional[ET._Element]) -> List[str]:
    """Collect <entry key='0' value='...'>, <entry key='1' value='...'> … under cfg."""
    out: List[str] = []
    if cfg is None:
        return out
    for k, v in iter_entries(cfg):
        if k.isdigit() and v:
            out.append(v.strip())
    return out


def parse_row_agg_settings(node_dir: Optional[Path]) -> RowAggregatorSettings:
    """Parse the row aggregation settings from the settings.xml file."""
    if not node_dir:
        return RowAggregatorSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return RowAggregatorSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return RowAggregatorSettings()

    category_col = first(model, ".//*[local-name()='entry' and @key='categoryColumn']/@value") or None
    method = (first(model, ".//*[local-name()='entry' and @key='aggregationMethod']/@value") or "COUNT").strip().upper()

    # Aggregation columns: frequencyColumns/selected_Internals + manualFilter/manuallySelected
    agg_cols: List[str] = []
    freq_el = first_el(model, ".//*[local-name()='config' and @key='frequencyColumns']")
    if freq_el is not None:
        sel_int = first_el(freq_el, ".//*[local-name()='config' and @key='selected_Internals']")
        agg_cols.extend(_collect_numeric_name_entries(sel_int))
        man_sel = first_el(freq_el, ".//*[local-name()='config' and @key='manualFilter']/*[local-name()='config' and @key='manuallySelected']")
        agg_cols.extend(_collect_numeric_name_entries(man_sel))
    # uniq while preserving order
    agg_cols = list(dict.fromkeys([c for c in agg_cols if c]))

    weight_col = first(model, ".//*[local-name()='entry' and @key='weightColumn']/@value") or None
    grand_totals = (first(model, ".//*[local-name()='entry' and @key='grandTotals']/@value") or "false").strip().lower() == "true"

    return RowAggregatorSettings(
        category_col=category_col or None,
        method=method or "COUNT",
        agg_cols=agg_cols,
        weight_col=weight_col or None,
        grand_totals=bool(grand_totals),
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    """Generate the necessary import statements for the output code."""
    return [
        "import pandas as pd",
        "import numpy as np",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.rowagg.RowAggregatorNodeFactory"
)


def _emit_row_agg_code(cfg: RowAggregatorSettings, node_id: str) -> List[str]:
    """
    Emit code that:
      - reads df from context
      - applies the selected aggregation (with optional grouping)
      - optionally produces 'grand totals' on port 2 when a category column is set & enabled
    """
    lines: List[str] = []

    # Serialize settings into python vars
    lines.append(f"_cat = {repr(cfg.category_col) if cfg.category_col else 'None'}")
    lines.append(f"_method = {repr((cfg.method or 'COUNT').upper())}")
    if cfg.agg_cols:
        cols = ", ".join(repr(c) for c in cfg.agg_cols)
        lines.append(f"_agg_cols = [{cols}]")
    else:
        lines.append("_agg_cols = []  # no explicit selection; will infer for some methods")
    lines.append(f"_wcol = {repr(cfg.weight_col) if cfg.weight_col else 'None'}")
    lines.append(f"_grand = {repr(bool(cfg.grand_totals))}")
    lines.append("")

    # Normalize/prepare selection
    lines.append("# Sanitize selections against incoming columns")
    lines.append("_cols_in = list(df.columns)")
    lines.append("_sel = [c for c in _agg_cols if c in _cols_in]")
    lines.append("")

    # COUNT: special behavior (rows per group)
    lines.append("if _method == 'COUNT':")
    lines.append("    if _cat and _cat in df.columns:")
    lines.append("        out_df = df.groupby(_cat, dropna=False).size().reset_index(name='OCCURRENCE_COUNT')")
    lines.append("        if _grand:")
    lines.append("            _grand_df = pd.DataFrame({'OCCURRENCE_COUNT': [len(df)]})")
    lines.append("    else:")
    lines.append("        out_df = pd.DataFrame({'OCCURRENCE_COUNT': [len(df)]})")
    lines.append("        _grand_df = None")
    lines.append("")
    lines.append("else:")
    lines.append("    # For numeric aggregations, ensure we have a column list")
    lines.append("    if not _sel:")
    lines.append("        # default to numeric/boolean columns excluding category/weight, if present")
    lines.append("        _ex = set([x for x in (_cat, _wcol) if x])")
    lines.append("        _sel = [c for c in _cols_in if c not in _ex]")
    lines.append("    num = df[_sel].apply(pd.to_numeric, errors='coerce')")
    lines.append("")
    lines.append("    if _cat and _cat in df.columns:")
    lines.append("        g = df[_cat]")
    lines.append("        if _method == 'SUM':")
    lines.append("            if _wcol and _wcol in df.columns:")
    lines.append("                w = pd.to_numeric(df[_wcol], errors='coerce').fillna(0.0)")
    lines.append("                out_df = (num.mul(w, axis=0)).groupby(g, dropna=False).sum(min_count=1)")
    lines.append("            else:")
    lines.append("                out_df = num.groupby(g, dropna=False).sum(min_count=1)")
    lines.append("        elif _method == 'AVERAGE':")
    lines.append("            if _wcol and _wcol in df.columns:")
    lines.append("                w = pd.to_numeric(df[_wcol], errors='coerce').fillna(0.0)")
    lines.append("                nume = (num.mul(w, axis=0)).groupby(g, dropna=False).sum(min_count=1)")
    lines.append("                den = w.groupby(g, dropna=False).sum(min_count=1)")
    lines.append("                out_df = nume.div(den.replace(0, np.nan), axis=0)")
    lines.append("            else:")
    lines.append("                out_df = num.groupby(g, dropna=False).mean()")
    lines.append("        elif _method == 'MINIMUM':")
    lines.append("            out_df = num.groupby(g, dropna=False).min()")
    lines.append("        elif _method == 'MAXIMUM':")
    lines.append("            out_df = num.groupby(g, dropna=False).max()")
    lines.append("        else:")
    lines.append("            raise ValueError(f'Unsupported aggregation method: {_method!r}')")
    lines.append("        out_df = out_df.reset_index()  # keep category column")
    lines.append("")
    lines.append("        # Grand totals if requested: same aggregation without grouping")
    lines.append("        _grand_df = None")
    lines.append("        if _grand:")
    lines.append("            if _method == 'SUM':")
    lines.append("                if _wcol and _wcol in df.columns:")
    lines.append("                    w = pd.to_numeric(df[_wcol], errors='coerce').fillna(0.0)")
    lines.append("                    _grand_df = (num.mul(w, axis=0)).sum(min_count=1).to_frame().T")
    lines.append("                else:")
    lines.append("                    _grand_df = num.sum(min_count=1).to_frame().T")
    lines.append("            elif _method == 'AVERAGE':")
    lines.append("                if _wcol and _wcol in df.columns:")
    lines.append("                    w = pd.to_numeric(df[_wcol], errors='coerce').fillna(0.0)")
    lines.append("                    nume = (num.mul(w, axis=0)).sum(min_count=1)")
    lines.append("                    den = w.sum()")
    lines.append("                    _grand_df = (nume / (den if den != 0 else np.nan)).to_frame().T")
    lines.append("                else:")
    lines.append("                    _grand_df = num.mean().to_frame().T")
    lines.append("            elif _method == 'MINIMUM':")
    lines.append("                _grand_df = num.min().to_frame().T")
    lines.append("            elif _method == 'MAXIMUM':")
    lines.append("                _grand_df = num.max().to_frame().T")
    lines.append("")
    lines.append("    else:")
    lines.append("        # No category: produce single-row results")
    lines.append("        if _method == 'SUM':")
    lines.append("            if _wcol and _wcol in df.columns:")
    lines.append("                w = pd.to_numeric(df[_wcol], errors='coerce').fillna(0.0)")
    lines.append("                out_df = (num.mul(w, axis=0)).sum(min_count=1).to_frame().T")
    lines.append("            else:")
    lines.append("                out_df = num.sum(min_count=1).to_frame().T")
    lines.append("        elif _method == 'AVERAGE':")
    lines.append("            if _wcol and _wcol in df.columns:")
    lines.append("                w = pd.to_numeric(df[_wcol], errors='coerce').fillna(0.0)")
    lines.append("                nume = (num.mul(w, axis=0)).sum(min_count=1)")
    lines.append("                den = w.sum()")
    lines.append("                out_df = (nume / (den if den != 0 else np.nan)).to_frame().T")
    lines.append("            else:")
    lines.append("                out_df = num.mean().to_frame().T")
    lines.append("        elif _method == 'MINIMUM':")
    lines.append("            out_df = num.min().to_frame().T")
    lines.append("        elif _method == 'MAXIMUM':")
    lines.append("            out_df = num.max().to_frame().T")
    lines.append("        else:")
    lines.append("            raise ValueError(f'Unsupported aggregation method: {_method!r}')")
    lines.append("        _grand_df = None  # no grouping → no separate grand totals table")
    lines.append("")
    # Publish
    lines.append(f"context['{node_id}:1'] = out_df")
    lines.append("if '_grand_df' in locals() and _grand_df is not None:")
    lines.append(f"    context['{node_id}:2'] = _grand_df")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,  # ignored (fixed ports: 1 + optional 2)
) -> List[str]:
    """Generate the Python body for the row aggregator node."""
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_row_agg_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # One table input
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    # Compute aggregation(s)
    lines.extend(_emit_row_agg_code(cfg, node_id))
    return lines


def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    """Generate the code for the Jupyter notebook cell."""
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Entry for emitters:
      - returns (imports, body_lines) if this module can handle the node
      - returns None otherwise
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    node_lines = generate_py_body(nid, npath, in_ports)

    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
