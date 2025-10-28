#!/usr/bin/env python3

"""Evaluates a KNIME Math Formula expression on an input table.

Overview
----------------------------
This module emits Python code that evaluates a KNIME Math Formula expression
on an input table and writes the result to the context. It translates the
KNIME/JEP expression to a Python expression, replacing column references and
mapping common functions.

Runtime Behavior
----------------------------
Inputs:
- Reads a single DataFrame from the context using the key format
  `context['<source_id>:<in_port>']`.

Outputs:
- Writes the resulting DataFrame back to the context with the key format
  `context['<node_id>:<out_port>']`, where the output is either a new column
  or an existing column based on the configuration.

Key algorithms or mappings:
- Translates KNIME/JEP expressions to Python, mapping operators and functions
  accordingly.

Edge Cases
----------------------------
The code handles cases such as empty or constant columns, NaNs, and ensures
that the output is appropriately typed based on the configuration.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries:
- pandas
- numpy

These dependencies are required by the generated code, not by this module.

Usage
----------------------------
This module is typically invoked by the knime2py emitter, which generates
Python code for KNIME nodes. An example of expected context access is:
```python
df = context['source_id:1']  # input table
```

Node Identity
----------------------------
KNIME factory id:
- FACTORY = "org.knime.ext.jep.JEPNodeFactory"

Configuration
----------------------------
The module uses the `MathFormulaSettings` dataclass for settings, which
includes the following important fields:
- `append`: Whether to append a new column (default: False).
- `replace_col`: The name of the column to replace (default: None).
- `expression`: The KNIME/JEP expression to evaluate (default: "").
- `convert_to_int`: Whether to convert the result to integer (default: False).
- `new_col_name`: The name of the new column when appending (default: "Math Formula").

The `parse_math_settings` function extracts these values from the settings.xml
file using XPath queries and provides fallbacks where necessary.

Limitations
----------------------------
Advanced JEP functions/operators not listed in the mappings are not translated.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.ext.jep/latest/
org.knime.ext.jep.JEPNodeFactory
"""

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
    """
    Convert a string value to a boolean.

    Args:
        v (Optional[str]): The string value to convert.
        default (bool): The default boolean value to return if v is None.

    Returns:
        bool: The converted boolean value.
    """
    if v is None:
        return default
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def parse_math_settings(node_dir: Optional[Path]) -> MathFormulaSettings:
    """
    Parse the settings.xml file to extract MathFormulaSettings.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        MathFormulaSettings: The parsed settings.
    """
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
    Translate a subset of KNIME/JEP to Python.

    Args:
        expr (str): The KNIME/JEP expression to translate.

    Returns:
        str: The translated Python expression.
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
    """
    Generate the necessary import statements for the Python code.

    Returns:
        List[str]: A list of import statements.
    """
    return ["import pandas as pd", "import numpy as np"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.ext.jep/latest/"
    "org.knime.ext.jep.JEPNodeFactory"
)

def _emit_math_code(cfg: MathFormulaSettings) -> List[str]:
    """
    Emit Python code that evaluates the translated expression and writes the result to the target column.

    Args:
        cfg (MathFormulaSettings): The configuration settings for the math formula.

    Returns:
        List[str]: The generated Python code lines.
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
    """
    Generate the body of the Python code for the node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: The generated Python code lines.
    """
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
    """
    Generate the code for a Jupyter notebook cell.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        str: The generated code for the notebook cell.
    """
    return "\n".join(generate_py_body(node_id, node_dir, in_ports, out_ports)) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handle the node and return the imports and body lines.

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
