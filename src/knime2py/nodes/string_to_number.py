#!/usr/bin/env python3

"""
String to Number conversion module.

Overview
----------------------------
This module generates Python code to convert selected string columns in a DataFrame to numeric
types based on settings parsed from a KNIME node's settings.xml file. It fits into the knime2py
generator pipeline by providing a mechanism to handle string-to-number conversions, ensuring
data integrity and type safety.

Runtime Behavior
----------------------------
Inputs:
- The generated code reads a DataFrame from the context using the key format `context['{src_id}:{in_port}']`.

Outputs:
- The converted DataFrame is written back to the context with the key format `context['{node_id}:{port}']`,
  where `port` is determined by the outgoing connections.

Key algorithms or mappings:
- The module supports custom decimal and thousands separators, inferring target types based on
  the cell class specified in the settings. It handles errors based on the `fail_on_error` flag,
  allowing for flexible data parsing.

Edge Cases
----------------------------
The code implements safeguards against empty or constant columns, NaNs, and ensures that
non-integer values are not cast to integer types when `fail_on_error` is set to True. It also
provides fallback paths for column selection.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries: pandas. These dependencies are
necessary for the execution of the generated code, not for this module itself.

Usage
----------------------------
This module is typically invoked by the knime2py emitter when processing a KNIME workflow. An
example of expected context access is:
```python
df = context['{src_id}:{in_port}']  # input table
```

Node Identity
----------------------------
KNIME factory id:
- FACTORY = "org.knime.base.node.preproc.colconvert.stringtonumber2.StringToNumber2NodeFactory"

Configuration
----------------------------
The `StringToNumberSettings` dataclass is used for settings, with important fields:
- `columns`: List of columns to convert.
- `decimal_sep`: Character used as the decimal separator (default: ".").
- `thousands_sep`: Character used as the thousands separator (default: None).
- `target_dtype`: Desired output type ("Float64" or "Int64", default: "Float64").
- `fail_on_error`: Flag to raise errors on parsing issues (default: False).

The `parse_string_to_number_settings` function extracts these values from the settings.xml file
using XPath queries, providing fallbacks where necessary.

Limitations
----------------------------
This module does not support all possible configurations available in KNIME and may approximate
some behaviors. Users should verify the output for edge cases.

References
----------------------------
For more information, refer to the KNIME documentation and the following hub URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.node.preproc.colconvert.stringtonumber2.StringToNumber2NodeFactory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory for this node
FACTORY = "org.knime.base.node.preproc.colconvert.stringtonumber2.StringToNumber2NodeFactory"

# ---------------------------------------------------------------------
# settings.xml → StringToNumberSettings
# ---------------------------------------------------------------------

@dataclass
class StringToNumberSettings:
    columns: List[str]
    decimal_sep: str
    thousands_sep: Optional[str]
    target_dtype: str          # "Float64" | "Int64"
    fail_on_error: bool
    generic_parse: bool        # (currently unused but captured)

def _collect_included_names(root: ET._Element) -> List[str]:
    """
    Collects the included column names from the provided XML root element.

    Args:
        root (ET._Element): The root element of the XML document.

    Returns:
        List[str]: A list of included column names.
    """
    base = first_el(root, ".//*[local-name()='config' and @key='model']"
                          "/*[local-name()='config' and @key='include']"
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
        if name:
            cols.append(name)
    # de-dup, preserve order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _target_from_cell_class(cls: Optional[str]) -> str:
    """
    Determines the target data type based on the cell class.

    Args:
        cls (Optional[str]): The cell class as a string.

    Returns:
        str: The inferred target data type ("Float64" or "Int64").
    """
    s = (cls or "").lower()
    if "doublecell" in s or "double" in s or "float" in s:
        return "Float64"
    if "intcell" in s or "longcell" in s or "int" in s or "long" in s:
        return "Int64"
    # default to float
    return "Float64"

def parse_string_to_number_settings(node_dir: Optional[Path]) -> StringToNumberSettings:
    """
    Parses the settings.xml file to extract StringToNumberSettings.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        StringToNumberSettings: The parsed settings.
    """
    if not node_dir:
        return StringToNumberSettings([], ".", None, "Float64", False, False)

    sp = node_dir / "settings.xml"
    if not sp.exists():
        return StringToNumberSettings([], ".", None, "Float64", False, False)

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")

    cols = _collect_included_names(root)

    dec = "."
    thou: Optional[str] = None
    if model is not None:
        dec = first(model, ".//*[local-name()='entry' and @key='decimal_separator']/@value") or "."
        raw_th = first(model, ".//*[local-name()='entry' and @key='thousands_separator']/@value")
        thou = (raw_th or "").strip() or None

        cell_cls = first(model, ".//*[local-name()='config' and @key='parse_type']/*[local-name()='entry' and @key='cell_class']/@value")
        target = _target_from_cell_class(cell_cls)

        fail = (first(model, ".//*[local-name()='entry' and @key='fail_on_error']/@value") or "false").strip().lower() in {"true", "1", "yes", "y"}
        genp = (first(model, ".//*[local-name()='entry' and @key='generic_parse']/@value") or "false").strip().lower() in {"true", "1", "yes", "y"}
    else:
        target, fail, genp = "Float64", False, False

    return StringToNumberSettings(
        columns=cols,
        decimal_sep=dec,
        thousands_sep=thou,
        target_dtype=target,
        fail_on_error=fail,
        generic_parse=genp,
    )

# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    """
    Generates the necessary imports for the conversion code.

    Returns:
        List[str]: A list of import statements.
    """
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.colconvert.stringtonumber2.StringToNumber2NodeFactory"
)

def _emit_convert_code(cfg: StringToNumberSettings) -> List[str]:
    """
    Emits the conversion code based on the provided settings.

    Args:
        cfg (StringToNumberSettings): The settings for the conversion.

    Returns:
        List[str]: The lines of code for the conversion.
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")

    # Target columns present in df
    if cfg.columns:
        joined = ", ".join(repr(c) for c in cfg.columns)
        lines.append(f"_target_cols = [{joined}]")
        lines.append("_cols = [c for c in _target_cols if c in out_df.columns]")
    else:
        lines.append("_cols = [c for c in out_df.columns]  # fallback: all columns")

    lines.append(f"_dec = {repr(cfg.decimal_sep or '.')}")
    lines.append(f"_thou = {repr(cfg.thousands_sep) if cfg.thousands_sep else 'None'}")
    lines.append(f"_target = {repr(cfg.target_dtype)}  # 'Float64' or 'Int64'")
    lines.append(f"_raise = {'True' if cfg.fail_on_error else 'False'}")

    lines.append("for _c in _cols:")
    # Prepare the string series
    lines.append("    _s = out_df[_c].astype('string').str.strip()")
    # Remove thousands separator if configured
    lines.append("    if _thou:")
    lines.append("        _s = _s.str.replace(_thou, '', regex=False)")
    # Normalize decimal separator to '.'
    lines.append("    if _dec and _dec != '.':")
    lines.append("        _s = _s.str.replace(_dec, '.', regex=False)")
    # Parse to numeric
    lines.append("    try:")
    lines.append("        _num = pd.to_numeric(_s, errors=('raise' if _raise else 'coerce'))")
    # Cast based on target dtype
    lines.append("        if _target == 'Int64':")
    lines.append("            if _raise:")
    lines.append("                # ensure no fractional part remains for non-NA values")
    lines.append("                if (_num.dropna() % 1 != 0).any():")
    lines.append("                    raise ValueError(f'Non-integer value encountered in column {_c}')")
    lines.append("                out_df[_c] = _num.astype('Int64')")
    lines.append("            else:")
    lines.append("                # round before casting to nullable Int64; preserves <NA>")
    lines.append("                out_df[_c] = _num.round().astype('Int64')")
    lines.append("        else:")
    lines.append("            out_df[_c] = _num.astype('Float64')")
    lines.append("    except Exception:")
    lines.append("        if _raise:")
    lines.append("            raise")
    lines.append("        # fail_on_error == False → leave the original column unchanged")
    lines.append("        pass")

    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generates the Python body for the String to Number node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: The lines of code for the node's body.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_string_to_number_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Single input
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_convert_code(cfg))

    # Publish to context (default port '1')
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
    Generates the code for a Jupyter notebook cell.

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
    Handles the generation of imports and body lines for the String to Number node.

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

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
