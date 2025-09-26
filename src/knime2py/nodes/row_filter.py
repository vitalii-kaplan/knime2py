#!/usr/bin/env python3

####################################################################################################
#
# Row Filter
#
# Filters rows of the input table according to predicates parsed from KNIME settings.xml.
# The generated pandas code builds a boolean mask from the list of predicates, combines them
# with AND/OR (matchCriteria), optionally inverts for NON_MATCHING output, and writes the
# result to this node's context output port(s).
#
# Supported operators (heuristic mapping):
#   - IS_MISSING                  →  df[col].isna()
#   - IS_NOT_MISSING              →  df[col].notna()
#   - EQ, EQUAL(S), =             →  numeric compare when possible; otherwise string compare
#   - NE, NOT_EQUAL, <>, !=       →  numeric compare when possible; otherwise string compare
#   - GT,  GREATER, >             →  to_numeric(df[col]) >  to_numeric(value)
#   - GE,  GREATER_EQUAL, >=      →  to_numeric(df[col]) >= to_numeric(value)
#   - LT,  LESS, <                →  to_numeric(df[col]) <  to_numeric(value)
#   - LE,  LESS_EQUAL, <=         →  to_numeric(df[col]) <= to_numeric(value)
#   - CONTAINS                    →  df[col].astype('string').str.contains(value, case=True, na=False)
#   - STARTS_WITH / ENDS_WITH     →  df[col].astype('string').str.startswith/endswith(value, na=False)
#
# Notes:
#   - We read *only* the <entry key="value" .../> items under predicateValues, avoiding
#     KNIME’s typeIdentifier entries like "org.knime.core.data.def.LongCell".
#   - Column names are resolved robustly (case-insensitive and normalized by dropping
#     non-alphanumerics). Missing configured columns produce a neutral predicate:
#         * AND-mode: neutral = True
#         * OR-mode:  neutral = False
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (
    first,
    first_el,
    normalize_in_ports,
    collect_module_imports,
    split_out_imports,
)

# KNIME factory ID
FACTORY = "org.knime.base.node.preproc.filter.row3.RowFilterNodeFactory"

# --------------------------------------------------------------------------------
# settings.xml → RowFilterSettings
# --------------------------------------------------------------------------------

@dataclass
class Predicate:
    column: Optional[str] = None
    operator: Optional[str] = None
    values: List[str] = field(default_factory=list)

@dataclass
class RowFilterSettings:
    match_and: bool = True                 # True → AND, False → OR
    output_mode: str = "MATCHING"          # MATCHING | NON_MATCHING
    predicates: List[Predicate] = field(default_factory=list)


def _bool(s: Optional[str], default: bool) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def _collect_predicate_values(p_cfg: ET._Element) -> List[str]:
    """
    Extract KNIME predicate values from:
      .../config[@key='predicateValues']/config[@key='values']/config[@key='0'..]/entry[@key='value']/@value
    Returns a list of strings (we coerce to numeric at runtime when needed).
    """
    if p_cfg is None:
        return []
    xpath = (
        ".//*[local-name()='config' and @key='predicateValues']"
        "/*[local-name()='config' and @key='values']"
        "/*[local-name()='config']/*[local-name()='entry' and @key='value']/@value"
    )
    vals = [str(v) for v in p_cfg.xpath(xpath)]
    return vals


def parse_row_filter_settings(node_dir: Optional[Path]) -> RowFilterSettings:
    if not node_dir:
        return RowFilterSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return RowFilterSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")

    match_and = True
    output_mode = "MATCHING"
    preds: List[Predicate] = []

    if model_el is not None:
        crit = (first(model_el, ".//*[local-name()='entry' and @key='matchCriteria']/@value") or "AND").strip().upper()
        match_and = (crit == "AND")
        output_mode = (first(model_el, ".//*[local-name()='entry' and @key='outputMode']/@value") or "MATCHING").strip().upper()

        # iterate predicate blocks under .../predicates/<config key='0'..>
        for p_cfg in model_el.xpath(
            ".//*[local-name()='config' and @key='predicates']/*[local-name()='config']"
        ):
            col = first(p_cfg, ".//*[local-name()='config' and @key='column']"
                               "/*[local-name()='entry' and @key='selected']/@value")
            op = (first(p_cfg, ".//*[local-name()='entry' and @key='operator']/@value") or "").strip().upper()
            vals = _collect_predicate_values(p_cfg)
            preds.append(Predicate(column=col or None, operator=op or None, values=vals))

    return RowFilterSettings(match_and=match_and, output_mode=output_mode, predicates=preds)


# --------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------

def generate_imports():
    # Need pandas and 're' for column normalization in runtime helpers
    return ["import pandas as pd", "import re as _re"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.filter.row3.RowFilterNodeFactory"
)


def _emit_filter_code(cfg: RowFilterSettings) -> List[str]:
    """
    Build pandas code that:
      - starts mask as all True (AND) or all False (OR)
      - for each predicate, computes cond_i and combines into mask
      - inverts for NON_MATCHING if requested
      - resolves column names robustly and coerces numeric comparisons at runtime
    """
    lines: List[str] = []

    # --- Column resolution & numeric coercion helpers (emitted into the cell) ---
    lines += [
        "def _rf_norm_name(s):",
        "    return _re.sub(r'[^a-z0-9]+', '', str(s).lower())",
        "_RF_LCMAP = {c.lower(): c for c in df.columns}",
        "_RF_NORMMAP = {_rf_norm_name(c): c for c in df.columns}",
        "",
        "def _rf_resolve(name):",
        "    if name in df.columns:",
        "        return name",
        "    if name is None:",
        "        return None",
        "    c = _RF_LCMAP.get(str(name).lower())",
        "    if c is not None:",
        "        return c",
        "    key = _rf_norm_name(name)",
        "    c = _RF_NORMMAP.get(key)",
        "    if c is not None:",
        "        return c",
        "    # Heuristic aliases for common aggregate column names",
        "    if key in {'occurrencecount','count','rowcount'}:",
        "        cand = [col for col in df.columns if 'count' in _rf_norm_name(col)]",
        "        if len(cand) == 1:",
        "            return cand[0]",
        "    return None",
        "",
        "def _rf_to_num(val):",
        "    s = pd.Series([val])",
        "    v = pd.to_numeric(s, errors='coerce').iloc[0]",
        "    return v",
        "",
    ]

    # Init mask depending on AND/OR
    if cfg.match_and:
        lines.append("mask = pd.Series(True, index=df.index)")
        comb = "&"
        neutral = "True"
    else:
        lines.append("mask = pd.Series(False, index=df.index)")
        comb = "|"
        neutral = "False"

    if not cfg.predicates:
        lines += ["# No predicates found; passthrough.", "out_df = df.copy()"]
        return lines

    # Map many KNIME op strings to canonical tokens
    def canon(op_raw: str) -> str:
        op = (op_raw or "").strip().upper()
        op = op.replace(" ", "_").replace("-", "_")
        # strip parentheses variants
        for ch in "()[]":
            op = op.replace(ch, "")
        mapping = {
            # equality
            "EQ": "EQ", "=": "EQ", "EQUAL": "EQ", "EQUALS": "EQ",
            # not equal
            "NE": "NE", "NEQ": "NE", "NOT_EQUAL": "NE", "!=": "NE", "<>": "NE",
            # greater / ge
            "GT": "GT", "GREATER": "GT", "GREATERTHAN": "GT", ">": "GT", "GREATER_THAN": "GT",
            "GE": "GE", "GREATEREQUAL": "GE", "GREATER_OR_EQUAL": "GE", ">=": "GE", "GREATER_EQUALS": "GE",
            # less / le
            "LT": "LT", "LESS": "LT", "LESSTHAN": "LT", "<": "LT", "LESS_THAN": "LT",
            "LE": "LE", "LESSEQUAL": "LE", "LESS_OR_EQUAL": "LE", "<=": "LE",
            # other
            "CONTAINS": "CONTAINS",
            "STARTS_WITH": "STARTS_WITH",
            "ENDS_WITH": "ENDS_WITH",
            "IS_MISSING": "IS_MISSING",
            "IS_NOT_MISSING": "IS_NOT_MISSING",
            "NOT_EQUAL_NOR_MISSING": "NOT_EQUAL_NOR_MISSING",
        }
        return mapping.get(op, op)

    # Emit each predicate
    for i, p in enumerate(cfg.predicates):
        col = p.column or ""
        op = canon(p.operator or "")
        vals = p.values or []

        s_var = f"_s{i}"
        c_var = f"_c{i}"
        lines.append(f"_col{i} = _rf_resolve({repr(col)})")
        lines.append(f"if _col{i} is None:")
        lines.append(f"    {c_var} = pd.Series({neutral}, index=df.index)  # missing column → neutral")
        lines.append("else:")
        lines.append(f"    {s_var} = df[_col{i}]")

        indent = "    "

        if op == "IS_MISSING":
            lines.append(f"{indent}{c_var} = {s_var}.isna()")
        elif op == "IS_NOT_MISSING":
            lines.append(f"{indent}{c_var} = {s_var}.notna()")

        elif op in {"EQ", "NE"}:
            # Decide numeric/boolean vs string comparison at runtime
            lines.append(f"{indent}_vals = {repr(vals)}")
            lines.append(f"{indent}_vals_num = [_rf_to_num(v) for v in _vals]")
            lines.append(f"{indent}_all_num = all(pd.notna(x) for x in _vals_num)")
            lines.append(f"{indent}_s_num = pd.to_numeric({s_var}, errors='coerce')")
            lines.append(f"{indent}_is_numlike = (pd.api.types.is_numeric_dtype(_s_num) or pd.api.types.is_bool_dtype(_s_num))")
            if op == "EQ":
                lines.append(f"{indent}if _all_num and _is_numlike:")
                if len(vals) <= 1:
                    rhs = "_vals_num[0] if _vals_num else float('nan')"
                    lines.append(f"{indent}    {c_var} = (_s_num == {rhs})")
                else:
                    lines.append(f"{indent}    {c_var} = _s_num.isin(_vals_num)")
                lines.append(f"{indent}else:")
                if len(vals) <= 1:
                    rhs = repr(vals[0]) if vals else "None"
                    lines.append(f"{indent}    {c_var} = ({s_var}.astype('string') == str({rhs})).fillna(False)")
                else:
                    lines.append(f"{indent}    {c_var} = {s_var}.astype('string').isin([str(v) for v in _vals]).fillna(False)")
            else:  # NE
                lines.append(f"{indent}if _all_num and _is_numlike:")
                if len(vals) <= 1:
                    rhs = "_vals_num[0] if _vals_num else float('nan')"
                    lines.append(f"{indent}    {c_var} = (_s_num != {rhs})")
                else:
                    lines.append(f"{indent}    {c_var} = ~_s_num.isin(_vals_num)")
                lines.append(f"{indent}else:")
                if len(vals) <= 1:
                    rhs = repr(vals[0]) if vals else "None"
                    lines.append(f"{indent}    {c_var} = ({s_var}.astype('string') != str({rhs})).fillna(False)")
                else:
                    lines.append(f"{indent}    {c_var} = ~{s_var}.astype('string').isin([str(v) for v in _vals]).fillna(False)")

        elif op in {"GT", "GE", "LT", "LE"}:
            rhs = repr(vals[0]) if vals else "None"
            comp = {
                "GT": ">",
                "GE": ">=",
                "LT": "<",
                "LE": "<=",
            }[op]
            lines.append(f"{indent}{c_var} = pd.to_numeric({s_var}, errors='coerce') {comp} _rf_to_num({rhs})")

        elif op == "CONTAINS":
            if len(vals) == 0:
                lines.append(f"{indent}{c_var} = pd.Series(False, index=df.index)")
            elif len(vals) == 1:
                lines.append(f"{indent}{c_var} = {s_var}.astype('string').str.contains({repr(vals[0])}, case=True, na=False)")
            else:
                ors = " | ".join([f"{s_var}.astype('string').str.contains({repr(v)}, case=True, na=False)" for v in vals])
                lines.append(f"{indent}{c_var} = ({ors})")
        elif op == "STARTS_WITH":
            if len(vals) == 0:
                lines.append(f"{indent}{c_var} = pd.Series(False, index=df.index)")
            elif len(vals) == 1:
                lines.append(f"{indent}{c_var} = {s_var}.astype('string').str.startswith({repr(vals[0])}, na=False)")
            else:
                ors = " | ".join([f"{s_var}.astype('string').str.startswith({repr(v)}, na=False)" for v in vals])
                lines.append(f"{indent}{c_var} = ({ors})")
        elif op == "ENDS_WITH":
            if len(vals) == 0:
                lines.append(f"{indent}{c_var} = pd.Series(False, index=df.index)")
            elif len(vals) == 1:
                lines.append(f"{indent}{c_var} = {s_var}.astype('string').str.endswith({repr(vals[0])}, na=False)")
            else:
                ors = " | ".join([f"{s_var}.astype('string').str.endswith({repr(v)}, na=False)" for v in vals])
                lines.append(f"{indent}{c_var} = ({ors})")
        else:
            # Unknown operator → neutral predicate
            lines.append(f"{indent}# unknown operator {repr(op)} → neutral")
            lines.append(f"{indent}{c_var} = pd.Series({neutral}, index=df.index)")

        # Combine
        lines.append(f"mask = (mask {comb} {c_var})")

    # Invert for NON_MATCHING
    lines.append(f"_invert = {repr(cfg.output_mode.upper() == 'NON_MATCHING')}")
    lines.append("final_mask = (~mask) if _invert else mask")
    lines.append("out_df = df[final_mask].copy()")

    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_row_filter_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Single input table
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    # Emit filter logic
    lines.extend(_emit_filter_code(cfg))

    # Publish result (default port 1)
    ports = out_ports or ["1"]
    for p in sorted({(p or '1') for p in ports}):
        lines.append(f"context['{node_id}:{p}'] = out_df")

    return lines


def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Returns (imports, body_lines) if this module can handle the node.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
