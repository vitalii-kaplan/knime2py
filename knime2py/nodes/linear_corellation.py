#!/usr/bin/env python3

####################################################################################################
#
# Linear Correlation
#
# Outputs (aligned to KNIME):
# - Port 1: Correlation measure (long table with exact KNIME column names)
#           Columns: "First column name", "Second column name",
#                    "Correlation value", "p value", "Degrees of freedom"
# - Port 2: Correlation matrix (wide Pearson correlation among selected numeric columns)
# - Port 3: Correlation model (dict with method, alternative, columns, and matrix)
#
# Settings honored (from settings.xml):
# - include-list: included_names / excluded_names + enforce_option (EnforceInclusion/EnforceExclusion)
# - pvalAlternative: TWO_SIDED | GREATER | LESS  (re-scales p from two-sided if SciPy is available)
# - columnPairsFilter: COMPATIBLE_PAIRS | ALL_PAIRS (we compute numeric↔numeric only)
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory
FACTORY = "org.knime.base.node.preproc.correlation.compute2.CorrelationCompute2NodeFactory"

# --------------------------------------------------------------------------------------------------
# settings.xml → Settings
# --------------------------------------------------------------------------------------------------

@dataclass
class CorrSettings:
    include_names: List[str]
    exclude_names: List[str]
    enforce_option: str            # "EnforceInclusion" | "EnforceExclusion"
    pval_alternative: str          # "TWO_SIDED" | "GREATER" | "LESS"
    pairs_filter: str              # "COMPATIBLE_PAIRS" | "ALL_PAIRS"
    possible_values_count: int     # kept for completeness

def _collect_name_list(root: ET._Element, key: str) -> List[str]:
    base = first_el(
        root,
        ".//*[local-name()='config' and @key='model']"
        "/*[local-name()='config' and @key='include-list']"
        f"/*[local-name()='config' and @key='{key}']"
    )
    if base is None:
        return []
    items: List[Tuple[int, str]] = []
    for k, v in iter_entries(base):
        if k.isdigit() and v is not None:
            try:
                items.append((int(k), v))
            except Exception:
                pass
    items.sort(key=lambda t: t[0])
    out: List[str] = []
    seen = set()
    for _, name in items:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out

def parse_corr_settings(node_dir: Optional[Path]) -> CorrSettings:
    defaults = CorrSettings(
        include_names=[],
        exclude_names=[],
        enforce_option="EnforceExclusion",
        pval_alternative="TWO_SIDED",
        pairs_filter="COMPATIBLE_PAIRS",
        possible_values_count=50,
    )

    if not node_dir:
        return defaults
    sp = node_dir / "settings.xml"
    if not sp.exists():
        return defaults

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")

    include_names = _collect_name_list(root, "included_names")
    exclude_names = _collect_name_list(root, "excluded_names")

    enforce_option = first(
        root,
        ".//*[local-name()='config' and @key='model']"
        "/*[local-name()='config' and @key='include-list']"
        "/*[local-name()='entry' and @key='enforce_option']/@value"
    ) or defaults.enforce_option
    enforce_option = enforce_option.strip()

    palt = first(model, ".//*[local-name()='entry' and @key='pvalAlternative']/@value") if model is not None else None
    palt = (palt or defaults.pval_alternative).strip().upper()

    pfilt = first(model, ".//*[local-name()='entry' and @key='columnPairsFilter']/@value") if model is not None else None
    pfilt = (pfilt or defaults.pairs_filter).strip().upper()

    pvc = first(model, ".//*[local-name()='entry' and @key='possibleValuesCount']/@value") if model is not None else None
    try:
        pvc_val = int(str(pvc).strip()) if pvc is not None else defaults.possible_values_count
    except Exception:
        pvc_val = defaults.possible_values_count

    return CorrSettings(
        include_names=include_names,
        exclude_names=exclude_names,
        enforce_option=enforce_option,
        pval_alternative=palt,
        pairs_filter=pfilt,
        possible_values_count=pvc_val,
    )

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    return ["import pandas as pd", "import numpy as np"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.correlation.compute2.CorrelationCompute2NodeFactory"
)

def _emit_corr_code(df_var: str, cfg: CorrSettings) -> List[str]:
    lines: List[str] = []
    lines.append("out_df = df.copy()  # passthrough copy (not strictly required)")
    lines.append("")
    lines.append("# Select numeric-like columns")
    lines.append(f"_num_like = {df_var}.select_dtypes(include=['number','Int64','Float64']).columns.tolist()")
    lines.append(f"_included_cfg = {repr(cfg.include_names or [])}")
    lines.append(f"_excluded_cfg = {repr(cfg.exclude_names or [])}")
    lines.append(f"_enforce = {repr(cfg.enforce_option or 'EnforceExclusion')}")
    lines.append("")
    lines.append("if _enforce == 'EnforceInclusion':")
    lines.append("    _cols = [c for c in _included_cfg if c in _num_like and c in out_df.columns]")
    lines.append("else:  # EnforceExclusion")
    lines.append("    _cols = [c for c in _num_like if c not in set(_excluded_cfg) and c in out_df.columns]")
    lines.append("")
    lines.append("# Guard: need at least two columns")
    lines.append("if len(_cols) < 2:")
    lines.append("    measure_out = pd.DataFrame(columns=[")
    lines.append("        'First column name', 'Second column name', 'Correlation value', 'p value', 'Degrees of freedom'")
    lines.append("    ])")
    lines.append("    matrix_out  = pd.DataFrame(index=pd.Index(_cols, name=None), columns=_cols, dtype='float64')")
    lines.append(f"    model_out   = {{'type':'CorrelationModel','method':'pearson','alternative':{repr(cfg.pval_alternative)},'columns':_cols,'matrix':matrix_out.copy()}}")
    lines.append("else:")
    lines.append("    _X = out_df[_cols].astype('float64')")
    lines.append("    matrix_out = _X.corr(method='pearson')")
    lines.append("")
    lines.append("    try:")
    lines.append("        from scipy import stats as _sstats  # optional")
    lines.append("        _have_scipy = True")
    lines.append("    except Exception:")
    lines.append("        _have_scipy = False")
    lines.append("")
    lines.append(f"    _alt = {repr(cfg.pval_alternative)}  # 'TWO_SIDED' | 'GREATER' | 'LESS'")
    lines.append("    rows = []")
    lines.append("    for i in range(len(_cols)):")
    lines.append("        for j in range(i+1, len(_cols)):")
    lines.append("            a, b = _cols[i], _cols[j]")
    lines.append("            pair = _X[[a, b]].dropna()")
    lines.append("            n = int(len(pair))")
    lines.append("            if n < 2:")
    lines.append("                r_val = np.nan; p_val = np.nan; dof = np.nan")
    lines.append("            else:")
    lines.append("                # degrees of freedom for Pearson's r")
    lines.append("                dof = n - 2")
    lines.append("                if _have_scipy:")
    lines.append("                    try:")
    lines.append("                        r_val, p_two = _sstats.pearsonr(pair[a].to_numpy(), pair[b].to_numpy())")
    lines.append("                    except Exception:")
    lines.append("                        r_val = pair[a].corr(pair[b])")
    lines.append("                        p_two = np.nan")
    lines.append("                else:")
    lines.append("                    r_val = pair[a].corr(pair[b])")
    lines.append("                    p_two = np.nan")
    lines.append("                # one-sided alternative remapping from two-sided (approx when available)")
    lines.append("                if np.isnan(p_two):")
    lines.append("                    p_val = np.nan")
    lines.append("                else:")
    lines.append("                    if _alt == 'TWO_SIDED':")
    lines.append("                        p_val = float(p_two)")
    lines.append("                    elif _alt == 'GREATER':")
    lines.append("                        p_val = float(p_two/2.0) if r_val >= 0 else float(1.0 - p_two/2.0)")
    lines.append("                    elif _alt == 'LESS':")
    lines.append("                        p_val = float(p_two/2.0) if r_val <= 0 else float(1.0 - p_two/2.0)")
    lines.append("                    else:")
    lines.append("                        p_val = float(p_two)")
    lines.append("            rows.append({")
    lines.append("                'First column name': a,")
    lines.append("                'Second column name': b,")
    lines.append("                'Correlation value': r_val,")
    lines.append("                'p value': p_val,")
    lines.append("                'Degrees of freedom': dof,")
    lines.append("            })")
    lines.append("    measure_out = pd.DataFrame(rows)")
    lines.append("")
    lines.append("    model_out = {")
    lines.append("        'type': 'CorrelationModel',")
    lines.append("        'method': 'pearson',")
    lines.append("        'alternative': _alt,")
    lines.append("        'columns': _cols,")
    lines.append("        'matrix': matrix_out.copy(),")
    lines.append("    }")
    lines.append("")
    lines.append("port1 = measure_out")
    lines.append("port2 = matrix_out")
    lines.append("port3 = model_out")
    return lines

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_corr_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0] if pairs else ("UNKNOWN", "1")
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_corr_code("df", cfg))

    # Publish (3 outputs expected)
    ports = [str(p or "1") for p in (out_ports or ["1", "2", "3"])]
    ports = list(dict.fromkeys(ports))
    while len(ports) < 3:
        ports.append(str(len(ports) + 1))
    ports = ports[:3]

    lines.append(f"context['{node_id}:{ports[0]}'] = port1")  # Correlation measure
    lines.append(f"context['{node_id}:{ports[1]}'] = port2")  # Correlation matrix
    lines.append(f"context['{node_id}:{ports[2]}'] = port3")  # Correlation model

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
    Returns (imports, body_lines) if this module can handle the node.
    """
    explicit_imports = collect_module_imports(generate_imports)

    # One input table
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])] or [("UNKNOWN", "1")]

    # Three outputs; use string port ids, not Edge objects
    out_port_ids = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2", "3"]

    node_lines = generate_py_body(nid, npath, in_ports, out_port_ids)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
