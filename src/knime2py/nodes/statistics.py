#!/usr/bin/env python3

####################################################################################################
#
# Statistics (Extended) — KNIME-like separation of "No. missings" vs "No. NaNs"
#
# Port 1: per-numeric-column stats with columns:
#   ['Column','Min','Max','Mean','Std. deviation','Variance','Skewness','Kurtosis',
#    'Overall sum','No. missings','No. NaNs','No. +∞s','No. -∞s','Median','Row count']
# Port 2: Nominal Histogram Table (Column, Values, Missing)
# Port 3: Occurrences Table (wide; one row per nominal column; columns are category labels)
#
# Notes:
# - "No. missings" counts true missing cells (nullable <NA>) where possible.
# - "No. NaNs" counts IEEE NaN values among non-missing entries.
#   For numpy float dtypes (no separate missing marker), we set missings=0 and NaNs=series.isnan().
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
    nominal_included: Optional[List[str]] = None
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
    return [
        "import pandas as pd",
        "import numpy as np",
        "from pandas.api.types import is_extension_array_dtype",
    ]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.stats/latest/"
    "org.knime.base.node.stats.viz.extended.ExtendedStatisticsNodeFactory"
)

def _emit_numeric_stats(df_var: str, include_median: bool) -> List[str]:
    """
    Emit KNIME-aligned numeric statistics for Port 1.
    - Separate 'No. missings' (nullable <NA>) from 'No. NaNs' (IEEE NaN).
    - Force float64 arrays for numeric computations & ±inf detection.
    """
    lines: List[str] = []
    lines.append(f"_num_cols = {df_var}.select_dtypes(include=['number','Int64','Float64']).columns.tolist()")
    lines.append("_rows = []")
    lines.append(f"_n_rows = int(len({df_var}))")
    lines.append("for _col in _num_cols:")
    lines.append(f"    _s_orig = {df_var}[_col]")
    lines.append("    # Robust numeric coercion for stats (NaN for non-coercibles)")
    lines.append("    _s = pd.to_numeric(_s_orig, errors='coerce')")
    lines.append("    _arr = _s.astype('float64').to_numpy(copy=False)")
    lines.append("")
    lines.append("    # ---- Missing vs NaN separation ----")
    lines.append("    _missings = 0  # true missing cells (<NA>)")
    lines.append("    _nans = 0      # IEEE NaN among non-missing values")
    lines.append("    if is_extension_array_dtype(_s_orig.dtype):")
    lines.append("        try:")
    lines.append("            _ea = _s_orig.array")
    lines.append("            _mask = getattr(_ea, '_mask', None)")
    lines.append("            if _mask is not None:")
    lines.append("                _mask = np.asarray(_mask, dtype=bool)")
    lines.append("                _missings = int(_mask.sum())")
    lines.append("                _data = getattr(_ea, '_data', None)")
    lines.append("                if _data is not None and np.issubdtype(_data.dtype, np.floating):")
    lines.append("                    _nans = int(np.isnan(_data[~_mask]).sum())")
    lines.append("                else:")
    lines.append("                    _nans = 0  # integer/boolean extension arrays cannot hold NaN")
    lines.append("            else:")
    lines.append("                # Extension array without mask (rare): treat all isna() as missing, NaNs unknown")
    lines.append("                _missings = int(_s_orig.isna().sum())")
    lines.append("                _nans = 0")
    lines.append("        except Exception:")
    lines.append("            # Conservative fallback")
    lines.append("            _missings = int(_s_orig.isna().sum())")
    lines.append("            _nans = int(np.isnan(_arr).sum())")
    lines.append("    else:")
    lines.append("        # Numpy-backed dtype: no separate missing marker; count NaNs only")
    lines.append("        _missings = 0")
    lines.append("        _nans = int(np.isnan(_arr).sum())")
    lines.append("")
    lines.append("    # ---- ±inf counts ----")
    lines.append("    _pos_inf = int(np.isposinf(_arr).sum())")
    lines.append("    _neg_inf = int(np.isneginf(_arr).sum())")
    lines.append("")
    lines.append("    # Finite subset for stats (exclude NaN and ±inf)")
    lines.append("    _finite = np.isfinite(_arr)")
    lines.append("    _s_fin = _s[_finite]")
    lines.append("    if _s_fin.empty:")
    lines.append("        _mn = _mx = _mean = _std = _var = _skew = _kurt = _sum = np.nan")
    if include_median:
        lines.append("        _median = np.nan")
    else:
        lines.append("        _median = None")
    lines.append("    else:")
    lines.append("        _mn = float(_s_fin.min())")
    lines.append("        _mx = float(_s_fin.max())")
    lines.append("        _mean = float(_s_fin.mean())")
    lines.append("        _std = float(_s_fin.std(ddof=0))")
    lines.append("        _var = float(_s_fin.var(ddof=0))")
    lines.append("        _skew = float(_s_fin.skew())")
    lines.append("        _kurt = float(_s_fin.kurt())")
    lines.append("        _sum = float(_s_fin.sum())")
    if include_median:
        lines.append("        _median = float(_s_fin.median())")
    else:
        lines.append("        _median = None")
    lines.append("")
    lines.append("    _row = {")
    lines.append("        'Column': _col,")
    lines.append("        'Min': _mn,")
    lines.append("        'Max': _mx,")
    lines.append("        'Mean': _mean,")
    lines.append("        'Std. deviation': _std,")
    lines.append("        'Variance': _var,")
    lines.append("        'Skewness': _skew,")
    lines.append("        'Kurtosis': _kurt,")
    lines.append("        'Overall sum': _sum,")
    lines.append("        'No. missings': _missings,")
    lines.append("        'No. NaNs': _nans,")
    lines.append("        'No. +∞s': _pos_inf,")
    lines.append("        'No. -∞s': _neg_inf,")
    lines.append("        'Median': _median,")
    lines.append("        'Row count': _n_rows,")
    lines.append("    }")
    lines.append("    _rows.append(_row)")
    lines.append("")
    lines.append("if _rows:")
    lines.append("    _order = ['Column','Min','Max','Mean','Std. deviation','Variance','Skewness','Kurtosis',"
                 "'Overall sum','No. missings','No. NaNs','No. +∞s','No. -∞s','Median','Row count']")
    lines.append("    stats_out = pd.DataFrame(_rows)[_order]")
    lines.append("else:")
    lines.append("    stats_out = pd.DataFrame(columns=['Column','Min','Max','Mean','Std. deviation','Variance',"
                 "'Skewness','Kurtosis','Overall sum','No. missings','No. NaNs','No. +∞s','No. -∞s','Median','Row count'])")
    return lines

def _emit_nominal_tables(df_var: str, selected_cols_expr: str, max_nom_out_expr: str) -> List[str]:
    lines: List[str] = []
    lines.append(f"_nom_sel = [c for c in {selected_cols_expr} if c in {df_var}.columns]")
    lines.append(f"if not _nom_sel:")
    lines.append(f"    _nom_sel = {df_var}.select_dtypes(include=['string','object','category']).columns.tolist()")
    # Port 2
    lines.append("hist_rows = []")
    lines.append("for _c in _nom_sel:")
    lines.append(f"    _s = {df_var}[_c]")
    lines.append("    _s_str = _s.astype('string')")
    lines.append("    _values  = int(_s_str.nunique(dropna=True))")
    lines.append("    _missing = int(_s_str.isna().sum())")
    lines.append("    hist_rows.append({'Column': _c, 'Values': _values, 'Missing': _missing})")
    lines.append("nom_hist_out = pd.DataFrame(hist_rows)")
    # Port 3
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
    lines.extend(_emit_numeric_stats(df_var, cfg.compute_median))
    sel = "[" + ", ".join(repr(c) for c in (cfg.nominal_included or [])) + "]"
    lines.extend(_emit_nominal_tables(df_var, sel, repr(int(cfg.max_nominal_out or 20))))
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

    ports = [str(p or "1") for p in (out_ports or ["1", "2", "3"])]
    ports = list(dict.fromkeys(ports))
    while len(ports) < 3:
        ports.append(str(len(ports) + 1))
    ports = ports[:3]

    lines.append(f"context['{node_id}:{ports[0]}'] = port1")
    lines.append(f"context['{node_id}:{ports[1]}'] = port2")
    lines.append(f"context['{node_id}:{ports[2]}'] = port3")

    return lines

def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    return "\n".join(generate_py_body(node_id, node_dir, in_ports, out_ports)) + "\n"

def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])] or [("UNKNOWN", "1")]
    out_port_ids = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2", "3"]
    node_lines = generate_py_body(nid, npath, in_ports, out_port_ids)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
