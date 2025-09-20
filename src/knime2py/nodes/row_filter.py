#!/usr/bin/env python3

####################################################################################################
#
# Row Filter
#
# Filters rows of the input table according to predicates parsed from KNIME settings.xml.
# The generated pandas code builds a boolean mask from the list of predicates, combines them
# with AND/OR (matchCriteria), optionally inverts for NON_MATCHING output, and writes the
# result to this node's context output port(s).
# Supported operators (heuristic mapping):
#   - IS_MISSING            →  df[col].isna()
#   - IS_NOT_MISSING        →  df[col].notna()
#   - EQUAL                 →  df[col] == value   (multiple values => isin(values))
#   - NOT_EQUAL             →  df[col] != value   (multiple values => ~isin(values))
#   - CONTAINS              →  df[col].astype('string').str.contains(value, case=True, na=False)
#   - STARTS_WITH           →  df[col].astype('string').str.startswith(value, na=False)
#   - ENDS_WITH             →  df[col].astype('string').str.endswith(value, na=False)
# Unknown operators fall back to a no-op for that predicate.
#
# - If a predicate expects values but none are provided, that predicate is skipped.
# - For EQUAL/NOT_EQUAL with multiple values, we use isin / ~isin.
# - For string ops we normalize via .astype('string') and guard NA with na=False.
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

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


def _collect_values(values_cfg: Optional[ET._Element]) -> List[str]:
    """
    Gather scalar values from a <config key='values'> subtree:
      <entry key='0' value='foo'/> <entry key='1' value='bar'/> ...
    """
    if values_cfg is None:
        return []
    vals: List[str] = []
    for k, v in iter_entries(values_cfg):
        # keep any string; ignore missing/None to avoid accidental "None" strings
        if v is not None:
            vals.append(v)
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

            vals_cfg = first_el(p_cfg, ".//*[local-name()='config' and @key='predicateValues']"
                                      "/*[local-name()='config' and @key='values']")
            vals = _collect_values(vals_cfg)

            preds.append(Predicate(column=col or None, operator=op or None, values=vals))

    return RowFilterSettings(match_and=match_and, output_mode=output_mode, predicates=preds)


# --------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------

def generate_imports():
    return ["import pandas as pd"]


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
    """
    lines: List[str] = []
    # Init mask depending on AND/OR
    if cfg.match_and:
        lines.append("mask = pd.Series(True, index=df.index)")
        comb = "&"
    else:
        lines.append("mask = pd.Series(False, index=df.index)")
        comb = "|"

    if not cfg.predicates:
        lines.append("# No predicates found; passthrough.")
        lines.append("out_df = df.copy()")
        return lines

    # Emit each predicate
    for i, p in enumerate(cfg.predicates):
        col = p.column or ""
        op = (p.operator or "").upper()
        vals = p.values or []

        # Skip invalid predicate (no column/op)
        if not col or not op:
            lines.append(f"# predicate {i}: skipped (missing column/operator)")
            continue

        # Define a safe series var
        s_var = f"_s{i}"
        lines.append(f"{s_var} = df[{repr(col)}]")

        # Produce condition depending on operator
        c_var = f"_c{i}"
        if op == "IS_MISSING":
            lines.append(f"{c_var} = {s_var}.isna()")
        elif op == "IS_NOT_MISSING":
            lines.append(f"{c_var} = {s_var}.notna()")
        elif op in {"EQUAL", "=="}:
            if len(vals) == 0:
                lines.append(f"{c_var} = pd.Series(False, index=df.index)  # no values to match")
            elif len(vals) == 1:
                v = vals[0]
                lines.append(f"{c_var} = ({s_var} == {repr(v)})")
            else:
                lines.append(f"{c_var} = {s_var}.isin({repr(vals)})")
        elif op in {"NOT_EQUAL", "!="}:
            if len(vals) == 0:
                lines.append(f"{c_var} = pd.Series(True, index=df.index)  # no values → always True")
            elif len(vals) == 1:
                v = vals[0]
                lines.append(f"{c_var} = ({s_var} != {repr(v)})")
            else:
                lines.append(f"{c_var} = ~{s_var}.isin({repr(vals)})")
        elif op == "CONTAINS":
            # Treat all values as OR within the predicate (any contains)
            if len(vals) == 0:
                lines.append(f"{c_var} = pd.Series(False, index=df.index)")
            elif len(vals) == 1:
                v = vals[0]
                lines.append(f"{c_var} = {s_var}.astype('string').str.contains({repr(v)}, case=True, na=False)")
            else:
                ors = " | ".join([f"{s_var}.astype('string').str.contains({repr(v)}, case=True, na=False)" for v in vals])
                lines.append(f"{c_var} = ({ors})")
        elif op == "STARTS_WITH":
            if len(vals) == 0:
                lines.append(f"{c_var} = pd.Series(False, index=df.index)")
            elif len(vals) == 1:
                v = vals[0]
                lines.append(f"{c_var} = {s_var}.astype('string').str.startswith({repr(v)}, na=False)")
            else:
                ors = " | ".join([f"{s_var}.astype('string').str.startswith({repr(v)}, na=False)" for v in vals])
                lines.append(f"{c_var} = ({ors})")
        elif op == "ENDS_WITH":
            if len(vals) == 0:
                lines.append(f"{c_var} = pd.Series(False, index=df.index)")
            elif len(vals) == 1:
                v = vals[0]
                lines.append(f"{c_var} = {s_var}.astype('string').str.endswith({repr(v)}, na=False)")
            else:
                ors = " | ".join([f"{s_var}.astype('string').str.endswith({repr(v)}, na=False)" for v in vals])
                lines.append(f"{c_var} = ({ors})")
        else:
            # Unknown operator → neutral predicate
            lines.append(f"# predicate {i}: unknown operator {repr(op)} → no-op")
            if cfg.match_and:
                lines.append(f"{c_var} = pd.Series(True, index=df.index)")
            else:
                lines.append(f"{c_var} = pd.Series(False, index=df.index)")

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
