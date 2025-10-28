#!/usr/bin/env python3

"""String Manipulation (Multi Column).

Overview
----------------------------
This module emits Python code that applies a subset of KNIME's Multi-Column String 
Manipulation to selected columns of a DataFrame, producing transformed columns based 
on specified expressions.

Runtime Behavior
----------------------------
Inputs:
- Reads a DataFrame from the context using the key format `context['{src_id}:{in_port}']`.

Outputs:
- Writes the transformed DataFrame back to the context with keys formatted as 
`context['{node_id}:{p}']`, where `p` is the port number.

Key algorithms or mappings:
- Supports nested replace operations, where the expression can include 
`replace(replace($$CURRENTCOLUMN$$,"Y","1"),"N","0")`.
- Selected columns are taken from the model's column selection in `settings.xml`.

Edge Cases
----------------------------
- Handles missing values by processing with pandas 'string' dtype to preserve NA.
- Implements an abort flag to control error handling during execution.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries: pandas. These 
dependencies are required for the generated code, not for this module.

Usage
----------------------------
Typically invoked by the KNIME emitter, this module is used in workflows that 
require string manipulation on multiple columns. An example of context access 
would be: `df = context['input_table:1']`.

Node Identity
----------------------------
KNIME factory id:
- FACTORY = "org.knime.base.node.preproc.stringmanipulation.multicolumn.MultiColumnStringManipulationNodeFactory".

Configuration
----------------------------
The settings are defined in the `MCStringSettings` dataclass, which includes:
- `columns`: List of selected columns for manipulation.
- `expression`: The string manipulation expression to apply.
- `mode`: Determines whether to append or replace columns (default is "REPLACE_COLUMNS").
- `suffix`: Suffix to append to new columns (default is "_transformed").
- `abort_on_error`: Flag to control error handling (default is True).
- `insert_missing_as_null`: Flag to control missing value handling (default is True).
The `parse_settings` function extracts these values from the `settings.xml` file.

Limitations
----------------------------
Currently, only the 'replace' function is supported; other string manipulation 
functions are not implemented.

References
----------------------------
For more information, refer to the KNIME documentation and the following hub URL:
https://hub.knime.com/knime/extensions/org.knime.features.javasnippet/latest/
org.knime.base.node.preproc.stringmanipulation.multicolumn.MultiColumnStringManipulationNodeFactory
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory
FACTORY = "org.knime.base.node.preproc.stringmanipulation.multicolumn.MultiColumnStringManipulationNodeFactory"

# --------------------------------------------------------------------------------
# settings.xml → Settings dataclass
# --------------------------------------------------------------------------------

@dataclass
class MCStringSettings:
    columns: List[str]
    expression: str
    mode: str                    # "APPEND_COLUMNS" | "REPLACE_COLUMNS"
    suffix: str                  # used only when APPEND_COLUMNS
    abort_on_error: bool
    insert_missing_as_null: bool


def _bool(s: Optional[str], default: bool) -> bool:
    """
    Convert a string representation of a boolean to a boolean value.

    Args:
        s (Optional[str]): The string to convert.
        default (bool): The default value to return if s is None.

    Returns:
        bool: The converted boolean value.
    """
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def _collect_included_names(root: ET._Element) -> List[str]:
    """
    Collect included column names from the XML configuration.

    Args:
        root (ET._Element): The root element of the XML configuration.

    Returns:
        List[str]: A list of included column names.
    """
    base = first_el(root, ".//*[local-name()='config' and @key='model']"
                          "/*[local-name()='config' and @key='column_selection']"
                          "/*[local-name()='config' and @key='included_names']")
    cols: List[str] = []
    if base is None:
        return cols
    numbered: List[tuple[int, str]] = []
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


def parse_settings(node_dir: Optional[Path]) -> MCStringSettings:
    """
    Parse the settings from the XML file located in the specified directory.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        MCStringSettings: The parsed settings.
    """
    if not node_dir:
        return MCStringSettings([], "", "REPLACE_COLUMNS", "_transformed", True, True)

    sp = node_dir / "settings.xml"
    if not sp.exists():
        return MCStringSettings([], "", "REPLACE_COLUMNS", "_transformed", True, True)

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")

    cols = _collect_included_names(root)

    expr = first(model, ".//*[local-name()='entry' and @key='EXPRESSION']/@value") if model is not None else ""
    expr = html.unescape(expr or "").strip()

    mode = first(model, ".//*[local-name()='entry' and @key='APPEND_OR_REPLACE']/@value") or "REPLACE_COLUMNS"
    mode = mode.strip().upper()

    suffix = first(model, ".//*[local-name()='entry' and @key='APPEND_COLUMN_SUFFIX']/@value") or "_transformed"

    abort_on_error = _bool(first(model, ".//*[local-name()='entry' and @key='Abort execution on evaluation errors']/@value"), True)
    insert_missing = _bool(first(model, ".//*[local-name()='entry' and @key='Insert missing values as null']/@value"), True)

    return MCStringSettings(
        columns=cols,
        expression=expr,
        mode=mode,
        suffix=suffix,
        abort_on_error=abort_on_error,
        insert_missing_as_null=insert_missing,
    )

# --------------------------------------------------------------------------------
# KNIME expression → list of operations (inner → outer)
# Supports nested replace(..., "...", "...")
# --------------------------------------------------------------------------------

# Simple recursive-descent parser for the subset:
#   Expr := Curr | Replace
#   Curr := '$$CURRENTCOLUMN$$'
#   Replace := 'replace(' Expr ',' String ',' String ')'
# Strings are double-quoted; inner quotes are already HTML-unescaped.

STR_RE = re.compile(r'\s*"([^"]*)"\s*', re.I)
CURR_TOKEN = "$$CURRENTCOLUMN$$"

def _parse_expr_ops(s: str) -> List[tuple[str, str, str]]:
    """
    Parse a KNIME expression into a sequence of ('replace', old, new) operations.

    Args:
        s (str): The KNIME expression to parse.

    Returns:
        List[tuple[str, str, str]]: A list of operations in evaluation order (inner → outer).
    """
    s = (s or "").strip()
    pos = 0
    n = len(s)

    def skip_ws():
        nonlocal pos
        while pos < n and s[pos].isspace():
            pos += 1

    def parse_curr() -> bool:
        nonlocal pos
        skip_ws()
        if s.startswith(CURR_TOKEN, pos):
            pos += len(CURR_TOKEN)
            return True
        return False

    def parse_string() -> Optional[str]:
        nonlocal pos
        skip_ws()
        m = STR_RE.match(s, pos)
        if not m:
            return None
        pos = m.end()
        return m.group(1)

    ops: List[tuple[str, str, str]] = []

    def parse_expr() -> bool:
        nonlocal pos, ops
        skip_ws()
        # replace(...) ?
        if s[pos:pos+8].lower() == "replace(":
            pos += 8
            # inner: either nested replace(...) or $$CURRENTCOLUMN$$
            if not (parse_expr() or parse_curr()):
                return False
            skip_ws()
            if pos >= n or s[pos] != ",":
                return False
            pos += 1
            old = parse_string()
            if old is None:
                return False
            skip_ws()
            if pos >= n or s[pos] != ",":
                return False
            pos += 1
            new = parse_string()
            if new is None:
                return False
            skip_ws()
            if pos >= n or s[pos] != ")":
                return False
            pos += 1
            # This replace applies AFTER the inner expression → append at end
            ops.append(("replace", old, new))
            return True
        # leaf
        return parse_curr()

    ok = parse_expr()
    skip_ws()
    if not ok or pos != n:
        return []  # unrecognized → identity
    # ops already collected inner→outer
    return ops

# --------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------

def generate_imports():
    """
    Generate the necessary imports for the generated code.

    Returns:
        List[str]: A list of import statements.
    """
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.javasnippet/latest/"
    "org.knime.base.node.preproc.stringmanipulation.multicolumn.MultiColumnStringManipulationNodeFactory"
)

def _emit_apply_code(cfg: MCStringSettings) -> List[str]:
    """
    Emit the code that applies the string manipulation based on the provided settings.

    Args:
        cfg (MCStringSettings): The settings for the string manipulation.

    Returns:
        List[str]: The lines of code that perform the string manipulation.
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    # Determine target columns present in df
    if cfg.columns:
        joined = ", ".join(repr(c) for c in cfg.columns)
        lines.append(f"_target_cols = [{joined}]")
        lines.append("_cols = [c for c in _target_cols if c in out_df.columns]")
    else:
        lines.append("_cols = [c for c in out_df.columns]  # fallback: all columns")

    # Translate expression → ops
    ops = _parse_expr_ops(cfg.expression or "")
    lines.append(f"_ops = {repr(ops)}  # parsed from EXPRESSION; [] = identity")

    lines.append("_append = " + ("True" if cfg.mode == "APPEND_COLUMNS" else "False"))
    lines.append(f"_suffix = {repr(cfg.suffix or '_transformed')}")

    # Apply per column
    if cfg.abort_on_error:
        lines.append("for _col in _cols:")
        lines.append("    _ser = out_df[_col].astype('string')")
        lines.append("    _val = _ser")
        lines.append("    for _op, _old, _new in _ops:")
        lines.append("        if _op == 'replace':")
        lines.append("            _val = _val.str.replace(_old, _new, regex=False)")
        lines.append("    _out_name = (_col + _suffix) if _append else _col")
        lines.append("    out_df[_out_name] = _val")
    else:
        lines.append("for _col in _cols:")
        lines.append("    try:")
        lines.append("        _ser = out_df[_col].astype('string')")
        lines.append("        _val = _ser")
        lines.append("        for _op, _old, _new in _ops:")
        lines.append("            if _op == 'replace':")
        lines.append("                _val = _val.str.replace(_old, _new, regex=False)")
        lines.append("        _out_name = (_col + _suffix) if _append else _col")
        lines.append("        out_df[_out_name] = _val")
        lines.append("    except Exception:")
        lines.append("        # skip column on error when abort==False")
        lines.append("        pass")

    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python body for the node based on the provided parameters.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: The lines of Python code for the node.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Single input table
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    # Emit transformation
    lines.extend(_emit_apply_code(cfg))

    # Publish result
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
    """
    Generate the code for a Jupyter notebook cell based on the node parameters.

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
    Handle the node and return the necessary imports and body lines.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming connections.
        outgoing: The outgoing connections.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines if the node can be handled; otherwise None.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
