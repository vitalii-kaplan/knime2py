#!/usr/bin/env python3

####################################################################################################
#
# Linear Correlation — KNIME order + nominal↔nominal pairs with p=0.0
#
# Port 1 (measure):
#   - Pair order follows source column order:
#       * If EnforceInclusion + included_names → exactly that order (filtered to present columns)
#       * Else → df.columns order
#   - COMPATIBLE_PAIRS: include numeric↔numeric and eligible nominal↔nominal
#   - For **nominal↔nominal pairs**, compute r on ordinal encoding and set **p value = 0.0**
#   - Numeric↔numeric p-values come from SciPy (if available), alternative remapping applied.
#
# Port 2 (matrix): numeric-only Pearson matrix in the same source order.
# Port 3 (model): metadata + the numeric matrix.
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

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
    possible_values_count: int

def _collect_name_list(root: ET._Element, key: str) -> List[str]:
    """
    Collects a list of unique names from the XML configuration based on the provided key.

    Args:
        root (ET._Element): The root element of the XML.
        key (str): The key to search for in the XML.

    Returns:
        List[str]: A list of unique names found in the XML configuration.
    """
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
    """
    Parses the correlation settings from the settings.xml file.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        CorrSettings: An instance of CorrSettings populated with values from the XML.
    """
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
    """
    Generates a list of necessary imports for the correlation computation.

    Returns:
        List[str]: A list of import statements.
    """
    return ["import pandas as pd", "import numpy as np"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.correlation.compute2.CorrelationCompute2NodeFactory"
)

def _emit_corr_code(df_var: str, cfg: CorrSettings) -> List[str]:
    """
    Emits the correlation computation code based on the provided DataFrame variable and settings.

    Args:
        df_var (str): The variable name of the input DataFrame.
        cfg (CorrSettings): The correlation settings.

    Returns:
        List[str]: A list of code lines for the correlation computation.
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()  # passthrough copy")
    lines.append("")
    lines.append("# 1) Type buckets and eligible nominal limit")
    lines.append(f"_pvc = int({int(cfg.possible_values_count)})")
    lines.append(f"_num_like_all = {df_var}.select_dtypes(include=['number','Int64','Float64','bool','boolean']).columns.tolist()")
    lines.append(f"_nom_cand_all = {df_var}.select_dtypes(include=['object','string','category']).columns.tolist()")
    lines.append("_nom_eligible_all = []")
    lines.append("for _c in _nom_cand_all:")
    lines.append(f"    _k = int({df_var}[_c].astype('string').nunique(dropna=True))")
    lines.append("    if _k <= _pvc:")
    lines.append("        _nom_eligible_all.append(_c)")
    lines.append("")
    lines.append("# 2) Establish the SOURCE ORDER")
    lines.append(f"_included_cfg = {repr(cfg.include_names or [])}")
    lines.append(f"_excluded_cfg = {repr(cfg.exclude_names or [])}")
    lines.append(f"_enforce = {repr(cfg.enforce_option or 'EnforceExclusion')}")
    lines.append("if _enforce == 'EnforceInclusion' and len(_included_cfg) > 0:")
    lines.append("    # Exact order from settings.xml (filtered to existing columns)")
    lines.append("    _order_source = [c for c in _included_cfg if c in out_df.columns]")
    lines.append("else:")
    lines.append("    # Input table order")
    lines.append(f"    _order_source = list({df_var}.columns)")
    lines.append("")
    lines.append("# 3) Apply include/exclude and compatibility (no reordering)")
    lines.append("if _enforce == 'EnforceInclusion' and len(_included_cfg) > 0:")
    lines.append("    _union = [c for c in _order_source if c in _included_cfg]")
    lines.append("else:  # EnforceExclusion")
    lines.append("    _union = [c for c in _order_source if c not in set(_excluded_cfg)]")
    lines.append("")
    lines.append("# Buckets in SOURCE order")
    lines.append("_num_like = [c for c in _union if c in _num_like_all]")
    lines.append("_nom_eligible = [c for c in _union if c in _nom_eligible_all]")
    lines.append("")
    lines.append("# 4) Build Port-1 measure rows in SOURCE order")
    lines.append(f"_pairs_mode = {repr(cfg.pairs_filter)}  # 'COMPATIBLE_PAIRS' | 'ALL_PAIRS'")
    lines.append("pairs = []")
    lines.append("for i in range(len(_union)):")
    lines.append("    for j in range(i+1, len(_union)):")
    lines.append("        a, b = _union[i], _union[j]")
    lines.append("        if _pairs_mode == 'COMPATIBLE_PAIRS':")
    lines.append("            # Keep order; only add if both numeric or both eligible nominal")
    lines.append("            if (a in _num_like and b in _num_like) or (a in _nom_eligible and b in _nom_eligible):")
    lines.append("                pairs.append((a, b))")
    lines.append("        else:  # ALL_PAIRS")
    lines.append("            pairs.append((a, b))")
    lines.append("")
    lines.append("# Early-empty check")
    lines.append("if len(pairs) == 0:")
    lines.append("    measure_out = pd.DataFrame(columns=[")
    lines.append("        'First column name', 'Second column name', 'Correlation value', 'p value', 'Degrees of freedom'")
    lines.append("    ])")
    lines.append("else:")
    lines.append("    try:")
    lines.append("        from scipy import stats as _sstats")
    lines.append("        _have_scipy = True")
    lines.append("    except Exception:")
    lines.append("        _have_scipy = False")
    lines.append(f"    _alt = {repr(cfg.pval_alternative)}")
    lines.append("")
    lines.append("    def _encode_series(s):")
    lines.append("        # Nominal → ordered integers with NaN preserved; numeric/bool → float64")
    lines.append("        if s.dtype.name in ('object','string') or str(s.dtype).startswith('category'):")
    lines.append("            _u = pd.Series(sorted(s.dropna().astype('string').unique()), dtype='string')")
    lines.append("            _map = {v:i for i,v in enumerate(_u.tolist())}")
    lines.append("            _enc = s.astype('string').map(_map)")
    lines.append("            return _enc.astype('float64')")
    lines.append("        return pd.to_numeric(s, errors='coerce').astype('float64')")
    lines.append("")
    lines.append("    rows = []")
    lines.append("    for a, b in pairs:")
    lines.append(f"        _sa_raw = {df_var}[a]")
    lines.append(f"        _sb_raw = {df_var}[b]")
    lines.append("        a_is_num = (a in _num_like)")
    lines.append("        b_is_num = (b in _num_like)")
    lines.append("        a_is_nom = (a in _nom_eligible)")
    lines.append("        b_is_nom = (b in _nom_eligible)")
    lines.append("        _sa = _encode_series(_sa_raw)")
    lines.append("        _sb = _encode_series(_sb_raw)")
    lines.append("        pair_df = pd.concat([_sa, _sb], axis=1).dropna()")
    lines.append("        n = int(len(pair_df))")
    lines.append("        if n < 2:")
    lines.append("            r_val = np.nan; p_val = np.nan; dof = np.nan")
    lines.append("        else:")
    lines.append("            dof = n - 2")
    lines.append("            # Correlation value (always Pearson on encoded data)")
    lines.append("            if _have_scipy and a_is_num and b_is_num:")
    lines.append("                try:")
    lines.append("                    r_val, p_two = _sstats.pearsonr(pair_df.iloc[:,0].to_numpy(), pair_df.iloc[:,1].to_numpy())")
    lines.append("                except Exception:")
    lines.append("                    r_val = pair_df.iloc[:,0].corr(pair_df.iloc[:,1])")
    lines.append("                    p_two = np.nan")
    lines.append("            else:")
    lines.append("                r_val = pair_df.iloc[:,0].corr(pair_df.iloc[:,1])")
    lines.append("                p_two = np.nan")
    lines.append("")
    lines.append("            # p-value policy:")
    lines.append("            # - numeric↔numeric: use SciPy p (with alternative remap) when available, else NaN")
    lines.append("            # - nominal↔nominal: KNIME shows p=0 → force 0.0")
    lines.append("            if a_is_nom and b_is_nom:")
    lines.append("                p_val = 0.0")
    lines.append("            elif np.isnan(p_two):")
    lines.append("                p_val = np.nan")
    lines.append("            else:")
    lines.append("                if _alt == 'TWO_SIDED':")
    lines.append("                    p_val = float(p_two)")
    lines.append("                elif _alt == 'GREATER':")
    lines.append("                    p_val = float(p_two/2.0) if r_val >= 0 else float(1.0 - p_two/2.0)")
    lines.append("                elif _alt == 'LESS':")
    lines.append("                    p_val = float(p_two/2.0) if r_val <= 0 else float(1.0 - p_two/2.0)")
    lines.append("                else:")
    lines.append("                    p_val = float(p_two)")
    lines.append("")
    lines.append("        rows.append({")
    lines.append("            'First column name': a,")
    lines.append("            'Second column name': b,")
    lines.append("            'Correlation value': r_val,")
    lines.append("            'p value': p_val,")
    lines.append("            'Degrees of freedom': dof,")
    lines.append("        })")
    lines.append("    measure_out = pd.DataFrame(rows)")
    lines.append("    # Do NOT sort — retain insertion order (matches KNIME)")
    lines.append("")
    lines.append("# 5) Port 2 — numeric Pearson matrix in SOURCE order")
    lines.append("_num_cols_ordered = [c for c in _union if c in _num_like]")
    lines.append("if len(_num_cols_ordered) >= 1:")
    lines.append(f"    _Xnum = {df_var}[_num_cols_ordered].apply(pd.to_numeric, errors='coerce').astype('float64')")
    lines.append("    matrix_out = _Xnum.corr(method='pearson')")
    lines.append("else:")
    lines.append("    matrix_out = pd.DataFrame(index=pd.Index(_num_cols_ordered, name=None), columns=_num_cols_ordered, dtype='float64')")
    lines.append("")
    lines.append("# 6) Port 3 — model")
    lines.append("model_out = {")
    lines.append("    'type': 'CorrelationModel',")
    lines.append("    'method': 'pearson',")
    lines.append("    'alternative': _alt if len(pairs) else 'TWO_SIDED',")
    lines.append("    'columns': _union,")
    lines.append("    'matrix': matrix_out.copy(),")
    lines.append("}")
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
    """
    Generates the Python body for the node, including imports and correlation computation.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: A list of code lines for the node's Python body.
    """
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

    lines.append(f"context['{node_id}:{ports[0]}'] = port1")  # measure (source order)
    lines.append(f"context['{node_id}:{ports[1]}'] = port2")  # matrix (numeric, source order)
    lines.append(f"context['{node_id}:{ports[2]}'] = port3")  # model

    return lines

def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    """
    Generates the code for a Jupyter notebook cell for the node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        str: The generated code as a string.
    """
    return "\n".join(generate_py_body(node_id, node_dir, in_ports, out_ports)) + "\n"

def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handles the node processing and returns the necessary imports and body lines.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming connections.
        outgoing: The outgoing connections.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines.
    """
    explicit_imports = collect_module_imports(generate_imports)

    # One input table
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])] or [("UNKNOWN", "1")]

    # Three outputs
    out_port_ids = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2", "3"]

    node_lines = generate_py_body(nid, npath, in_ports, out_port_ids)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
