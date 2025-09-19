#!/usr/bin/env python3

####################################################################################################
#
# Math Formula (JEP)
#
# Evaluates a KNIME Math Formula expression on an input table and writes the result to context.
# Parses settings.xml and emits pandas/numpy code that:
#   - translates KNIME/JEP expression to a Python expression
#   - replaces $col$ references with df[<col>]
#   - maps '^' (power) → '**', and common functions (ln/log/sqrt/exp/round/ceil/floor)
#   - optionally converts result to Int (round → Int64) if configured
#   - appends a new column or replaces an existing one based on settings
#
#   - Only a small set of functions is mapped (ln, log, log10, sqrt, exp, round, ceil, floor).
#   - Advanced JEP functions/operators not listed above are not translated.
#
####################################################################################################

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, normalize_in_ports, iter_entries, collect_module_imports, split_out_imports

# KNIME factory
FACTORY = "org.knime.ext.jep.JEPNodeFactory"

# --------------------------------------------------------------------------------------------------
# settings.xml → Settings
# --------------------------------------------------------------------------------------------------

@dataclass
class MathFormulaSettings:
    append: bool = False                  # entry key="append_column"
    replace_col: Optional[str] = None     # entry key="replaced_column" (used when append == False)
    expression: str = ""                  # entry key="expression"
    convert_to_int: bool = False          # entry key="convert_to_int"
    new_col_name: str = "Math Formula"    # fallback name when appending and no explicit is present


def _bool(v: Optional[str], default: bool) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def parse_math_settings(node_dir: Optional[Path]) -> MathFormulaSettings:
    if not node_dir:
        return MathFormulaSettings()
    sp = node_dir / "settings.xml"
    if not sp.exists():
        return MathFormulaSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")

    append = _bool(first(model, ".//*[local-name()='entry' and @key='append_column']/@value"), False) if model is not None else False
    replace_col = first(model, ".//*[local-name()='entry' and @key='replaced_column']/@value") if model is not None else None
    expr = first(model, ".//*[local-name()='entry' and @key='expression']/@value") if model is not None else ""
    expr = html.unescape(expr or "").strip()
    convert_to_int = _bool(first(model, ".//*[local-name()='entry' and @key='convert_to_int']/@value"), False) if model is not None else False

    # Some KNIME variants let you name the appended column; if not present, keep our default
    new_col_name = first(model, ".//*[local-name()='entry' and @key='new_column_name']/@value") or "Math Formula"

    return MathFormulaSettings(
        append=append,
        replace_col=(replace_col or None),
        expression=expr,
        convert_to_int=convert_to_int,
        new_col_name=new_col_name,
    )

# --------------------------------------------------------------------------------------------------
# KNIME/JEP expression → Python expression
# --------------------------------------------------------------------------------------------------

_COL_TOKEN = re.compile(r"\$(.+?)\$")  # non-greedy: $...$

def _translate_expression(expr: str) -> str:
    """
    Translate a subset of KNIME/JEP to Python:
      - $col$ → df[<repr col>]
      - '^' → '**'
      - ln(), log(), log10(), sqrt(), exp(), round(), ceil(), floor() mapped to numpy
    """
    s = (expr or "").strip()

    # 1) Replace $col$ tokens with df['col']
    def repl_col(m: re.Match) -> str:
        col = m.group(1)
        return f"df[{repr(col)}]"
    s = _COL_TOKEN.sub(repl_col, s)

    # 2) Map power operator BEFORE func-name replacements
    #    (Knime uses '^' for exponent; Python uses '**')
    s = s.replace("^", "**")

    # 3) Map common functions to numpy
    #    Use word-boundary to avoid touching column names inside df['...']
    func_maps = {
        r"\bln\(": "np.log(",
        r"\blog10\(": "np.log10(",
        r"\blog\(": "np.log(",
        r"\bsqrt\(": "np.sqrt(",
        r"\bexp\(": "np.exp(",
        r"\bround\(": "np.round(",
        r"\bceil\(": "np.ceil(",
        r"\bfloor\(": "np.floor(",
        # pow(a,b) works in Python; keep as-is (or map to np.power if you prefer)
    }
    for pat, repl in func_maps.items():
        s = re.sub(pat, repl, s)

    return s

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    return ["import pandas as pd", "import numpy as np"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.ext.jep/latest/"
    "org.knime.ext.jep.JEPNodeFactory"
)

def _emit_math_code(cfg: MathFormulaSettings) -> List[str]:
    """
    Emit Python that:
      - copies input df → out_df
      - evaluates translated expression into _res (Series or scalar broadcast)
      - optionally converts to Int64
      - writes to target column (append or replace)
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    py_expr = _translate_expression(cfg.expression or "")
    lines.append(f"_expr = {repr(py_expr)}  # translated from JEP")
    lines.append("# Evaluate the expression in a restricted namespace")
    lines.append("_ns = {'np': np, 'pd': pd, 'df': df}")
    lines.append("_res = eval(_expr, {'__builtins__': {}}, _ns)")

    # Convert to int if requested (match KNIME behavior approximately: round → Int64)
    if cfg.convert_to_int:
        lines.append("_res = pd.to_numeric(_res, errors='coerce').round().astype('Int64')")

    # Determine target column name
    if cfg.append:
        tgt = cfg.new_col_name or "Math Formula"
    else:
        tgt = cfg.replace_col or "Math Formula"
    lines.append(f"_target_col = {repr(tgt)}")
    lines.append("out_df[_target_col] = _res")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_math_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Single table input
    src_id, in_port = normalize_in_ports(in_ports)[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    # Transform
    lines.extend(_emit_math_code(cfg))

    # Publish to context
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
    return "\n".join(generate_py_body(node_id, node_dir, in_ports, out_ports)) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Return (imports, body_lines) if we can handle this node; otherwise None.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
