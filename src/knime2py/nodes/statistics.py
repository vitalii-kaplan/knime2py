#!/usr/bin/env python3

####################################################################################################
#
# Statistics (Extended)
#
# Outputs (aligned to KNIME):
# - Port 1: Statistics table (per-numeric column stats: Count, Missing, Unique, Mean, [Median], Std, Min, Max)
# - Port 2: Nominal Histogram Table (per nominal column; columns = Column, Values, Missing)
# - Port 3: Occurrences Table (wide; one row per nominal column; columns are category labels with counts)
#
# - compute_median: bool → include Median in numeric stats
# - filter_nominal_columns/included_names: list → which columns to treat as nominal
# - num_nominal-values_output: int → cap of categories per nominal column for Port 3 (occurrence table)
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

FACTORY = "org.knime.base.node.stats.viz.extended.ExtendedStatisticsNodeFactory"

# --------------------------------------------------------------------------------------------------
# settings.xml → Settings
# --------------------------------------------------------------------------------------------------

@dataclass
class StatsSettings:
    compute_median: bool = True
    nominal_included: List[str] = None
    max_nominal_out: int = 20  # per-column cap for occurrence table (wide)

def _bool(s: Optional[str], default: bool) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"true", "1", "yes", "y"}

def _collect_included_names(root: ET._Element) -> List[str]:
    base = first_el(
        root,
        ".//*[local-name()='config' and @key='model']"
        "/*[local-name()='config' and @key='filter_nominal_columns']"
        "/*[local-name()='config' and @key='included_names']"
    )
    cols: List[str] = []
    if base is None:
        return cols
    numbered: List[Tuple[int, str]] = []
    for k, v in iter_entries(base):
        if k.isdigit() and v is not None:
            try:
                numbered.append((int(k), v))
            except Exception:
                pass
    for _, name in sorted(numbered, key=lambda t: t[0]):
        cols.append(name)
    # de-dup preserving order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def parse_stats_settings(node_dir: Optional[Path]) -> StatsSettings:
    if not node_dir:
        return StatsSettings(True, [], 20)

    sp = node_dir / "settings.xml"
    if not sp.exists():
        return StatsSettings(True, [], 20)

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")

    compute_median = True
    max_nominal_out = 20
    if model is not None:
        compute_median = _bool(first(model, ".//*[local-name()='entry' and @key='compute_median']/@value"), True)
        # prefer the explicit "*_output" cap if present; fallback to generic "num_nominal-values"
        nout = first(model, ".//*[local-name()='entry' and @key='num_nominal-values_output']/@value")
        if not nout:
            nout = first(model, ".//*[local-name()='entry' and @key='num_nominal-values']/@value")
        try:
            if nout:
                max_nominal_out = max(1, int(str(nout).strip()))
        except Exception:
            pass

    included = _collect_included_names(root)

    return StatsSettings(
        compute_median=compute_median,
        nominal_included=included,
        max_nominal_out=max_nominal_out or 20,
    )

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    return ["import pandas as pd", "import numpy as np"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.stats/latest/"
    "org.knime.base.node.stats.viz.extended.ExtendedStatisticsNodeFactory"
)

def _emit_numeric_stats(df_var: str, include_median: bool) -> List[str]:
    lines: List[str] = []
    lines.append(f"_num_cols = {df_var}.select_dtypes(include=['number', 'Int64', 'Float64']).columns.tolist()")
    lines.append("if _num_cols:")
    agg_list = ["'count'", "'mean'", "'std'", "'min'", "'max'"]
    if include_median:
        agg_list.append("'median'")
    agg_str = ", ".join(agg_list)
    lines.append(f"    _num = {df_var}[_num_cols]")
    lines.append(f"    _stats = _num.agg([{agg_str}]).T")
    lines.append("    _stats['Missing'] = (_num.isna().sum()).astype('Int64')")
    lines.append("    _stats['Unique']  = (_num.nunique(dropna=True)).astype('Int64')")
    order = ["count", "Missing", "Unique", "mean", "std", "min", "max"]
    if include_median:
        order.insert(4, "median")  # after mean
    lines.append(f"    _order = {order!r}")
    lines.append("    _present = [c for c in _order if c in _stats.columns]")
    lines.append("    stats_out = _stats[_present].reset_index().rename(columns={'index': 'Column'})")
    lines.append("else:")
    empty_cols = ['Column','count','Missing','Unique','mean','std','min','max']
    if include_median:
        empty_cols.insert(4, 'median')
    lines.append(f"    stats_out = pd.DataFrame(columns={empty_cols!r})")
    return lines

def _emit_nominal_tables(df_var: str, selected_cols_expr: str, max_nom_out_expr: str) -> List[str]:
    """
    - Nominal Histogram Table → nom_hist_out (Column, Values, Missing)
    - Occurrences Table (wide) → occ_out (one row per nominal column; category columns with counts)
    """
    lines: List[str] = []
    # choose nominal columns (if none configured, fall back to object/string/category)
    lines.append(f"_nom_sel = [c for c in {selected_cols_expr} if c in {df_var}.columns]")
    lines.append(f"if not _nom_sel:")
    lines.append(f"    _nom_sel = {df_var}.select_dtypes(include=['string','object','category']).columns.tolist()")

    # Port 2: Nominal Histogram Table (Column, Values, Missing)
    lines.append("hist_rows = []")
    lines.append("for _c in _nom_sel:")
    lines.append(f"    _s = {df_var}[_c]")
    lines.append("    _s_str = _s.astype('string')")
    lines.append("    _values  = int(_s_str.nunique(dropna=True))  # number of distinct categories")
    lines.append("    _missing = int(_s_str.isna().sum())")
    lines.append("    hist_rows.append({'Column': _c, 'Values': _values, 'Missing': _missing})")
    lines.append("nom_hist_out = pd.DataFrame(hist_rows)")

    # Port 3: Occurrences Table (wide)
    lines.append("dist_cols_union: set = set()")
    lines.append("per_col_dicts: list = []")
    lines.append("for _c in _nom_sel:")
    lines.append(f"    _vc = {df_var}[_c].astype('string').value_counts(dropna=True).head({max_nom_out_expr})")
    lines.append("    _d = {str(k): int(v) for k, v in _vc.items()}")
    lines.append("    _row = {'Column': _c}")
    lines.append("    _row.update(_d)")
    lines.append("    per_col_dicts.append(_row)")
    lines.append("    dist_cols_union.update(_d.keys())")
    lines.append("if per_col_dicts:")
    lines.append("    _label_cols = sorted(dist_cols_union)")
    lines.append("    _cols = ['Column'] + _label_cols")
    lines.append("    occ_out = pd.DataFrame([{k: d.get(k, 0) for k in _cols} for d in per_col_dicts], columns=_cols)")
    lines.append("else:")
    lines.append("    occ_out = pd.DataFrame(columns=['Column'])")
    return lines

def _emit_code(df_var: str, cfg: StatsSettings) -> List[str]:
    lines: List[str] = []
    lines.append("out_df = df.copy()  # passthrough copy (not strictly needed)")
    # Port 1: Statistics table
    lines.extend(_emit_numeric_stats(df_var, cfg.compute_median))
    # Ports 2 & 3: Nominal Histogram + Occurrences (wide)
    sel = "[" + ", ".join(repr(c) for c in (cfg.nominal_included or [])) + "]"
    lines.extend(_emit_nominal_tables(df_var, sel, repr(int(cfg.max_nominal_out or 20))))
    # Bind named outputs
    lines.append("port1 = stats_out")
    lines.append("port2 = nom_hist_out")
    lines.append("port3 = occ_out")
    return lines

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_stats_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0] if pairs else ("UNKNOWN", "1")
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_code("df", cfg))

    # Publish to context (expect 3 outputs)
    ports = [str(p or "1") for p in (out_ports or ["1", "2", "3"])]
    ports = list(dict.fromkeys(ports))
    while len(ports) < 3:
        ports.append(str(len(ports) + 1))
    ports = ports[:3]

    lines.append(f"context['{node_id}:{ports[0]}'] = port1")  # Statistics table
    lines.append(f"context['{node_id}:{ports[1]}'] = port2")  # Nominal Histogram Table
    lines.append(f"context['{node_id}:{ports[2]}'] = port3")  # Occurrences Table

    return lines

def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    return "\n".join(generate_py_body(node_id, node_dir, in_ports, out_ports)) + "\n"

def handle(ntype, nid, npath, incoming, outgoing):
    """
    Returns (imports, body_lines) if this module can handle the node; else None.
    """
    explicit_imports = collect_module_imports(generate_imports)

    # One input table
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])] or [("UNKNOWN", "1")]

    # Three outputs; extract string port ids
    out_port_ids = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2", "3"]

    node_lines = generate_py_body(nid, npath, in_ports, out_port_ids)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
