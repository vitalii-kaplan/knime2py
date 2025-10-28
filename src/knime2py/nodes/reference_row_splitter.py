#!/usr/bin/env python3

"""
Reference Row Splitter.

Overview
----------------------------
This module generates Python code to split a data table into two outputs based on the 
membership of keys in a separate reference table. It fits into the knime2py generator 
pipeline by producing code that can be executed in a Python environment.

Runtime Behavior
----------------------------
Inputs:
- Reads two DataFrames from the context: one for the data table and one for the reference 
  table, identified by their respective keys.

Outputs:
- Writes two DataFrames to the context:
  - Port 1 (`context['{node_id}:1']`): rows from the data table whose keys match the 
    reference table (matching).
  - Port 2 (`context['{node_id}:2']`): rows from the data table whose keys do not match 
    the reference table (non-matching).

Key algorithms:
- The code coerces join keys to pandas 'string' dtype and ignores NaNs in the reference key 
  set. It raises a KeyError if a configured column is missing.

Edge Cases
----------------------------
The code handles cases where columns may be empty or constant, and it safeguards against 
NaNs in the reference key set. It raises appropriate errors for missing columns.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries: pandas. These dependencies 
are required for the generated code, not for this module.

Usage
----------------------------
This module is typically invoked by the knime2py emitter when generating code for a 
reference row splitter node. An example of expected context access is:
```python
df_left = context['source_id:1']  # Accessing the data table
df_right = context['source_id:2']  # Accessing the reference table
```

Node Identity
----------------------------
KNIME factory id: 
- FACTORY = "org.knime.base.node.preproc.filter.rowref.RowSplitRefNodeFactory"

Configuration
----------------------------
The settings are defined in the `RefRowSplitSettings` dataclass, which includes:
- `data_use_rowid`: Indicates if the data table uses row IDs (default: False).
- `data_col`: The name of the column in the data table to use as a key (default: None).
- `ref_use_rowid`: Indicates if the reference table uses row IDs (default: False).
- `ref_col`: The name of the column in the reference table to use as a key (default: None).

The `parse_refsplit_settings` function extracts these values from the settings.xml file 
using XPath queries, with fallbacks to defaults if necessary.

Limitations
----------------------------
This module does not support certain advanced configurations available in KNIME, and 
approximations may occur in behavior compared to the original KNIME node.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.node.preproc.filter.rowref.RowSplitRefNodeFactory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (
    first,
    first_el,
    normalize_in_ports,
    collect_module_imports,
    split_out_imports,
)

# KNIME factory
FACTORY = "org.knime.base.node.preproc.filter.rowref.RowSplitRefNodeFactory"

# -------------------------------------------------------------------------------------------------
# settings.xml → RefRowSplitSettings
# -------------------------------------------------------------------------------------------------

@dataclass
class RefRowSplitSettings:
    data_use_rowid: bool = False
    data_col: Optional[str] = None
    ref_use_rowid: bool = False
    ref_col: Optional[str] = None


def _bool(s: Optional[str], default: bool = False) -> bool:
    """
    Convert a string to a boolean value.

    Args:
        s (Optional[str]): The string to convert.
        default (bool): The default value to return if s is None.

    Returns:
        bool: The converted boolean value.
    """
    if s is None:
        return default
    return str(s).strip().lower() in {"true", "1", "yes", "y"}


def parse_refsplit_settings(node_dir: Optional[Path]) -> RefRowSplitSettings:
    """
    Parse the settings from the settings.xml file for the reference row splitter.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        RefRowSplitSettings: The parsed settings.
    """
    if not node_dir:
        return RefRowSplitSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return RefRowSplitSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return RefRowSplitSettings()

    data_cfg = first_el(model, ".//*[local-name()='config' and @key='dataTableColumn']")
    ref_cfg  = first_el(model, ".//*[local-name()='config' and @key='referenceTableColumn']")

    data_use_rowid = _bool(first(data_cfg, ".//*[local-name()='entry' and @key='useRowID']/@value"), False) if data_cfg is not None else False
    data_col = first(data_cfg, ".//*[local-name()='entry' and @key='columnName']/@value") if data_cfg is not None else None

    ref_use_rowid = _bool(first(ref_cfg,  ".//*[local-name()='entry' and @key='useRowID']/@value"), False) if ref_cfg is not None else False
    ref_col = first(ref_cfg,  ".//*[local-name()='entry' and @key='columnName']/@value") if ref_cfg is not None else None

    return RefRowSplitSettings(
        data_use_rowid=data_use_rowid,
        data_col=(data_col or None),
        ref_use_rowid=ref_use_rowid,
        ref_col=(ref_col or None),
    )

# -------------------------------------------------------------------------------------------------
# Code generators
# -------------------------------------------------------------------------------------------------

def generate_imports():
    """
    Generate the necessary imports for the Python code.

    Returns:
        List[str]: A list of import statements.
    """
    return ["import pandas as pd"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.filter.rowref.RowSplitRefNodeFactory"
)


def _emit_split_code(cfg: RefRowSplitSettings, node_id: str) -> List[str]:
    """
    Emit the code for splitting the data based on the configuration settings.

    Args:
        cfg (RefRowSplitSettings): The configuration settings for the splitter.
        node_id (str): The ID of the node.

    Returns:
        List[str]: The lines of code to perform the split.
    """
    lines: List[str] = []

    # Serialize config
    lines.append(f"_data_use_rowid = {('True' if cfg.data_use_rowid else 'False')}")
    lines.append(f"_data_col = {repr(cfg.data_col) if cfg.data_col else 'None'}")
    lines.append(f"_ref_use_rowid = {('True' if cfg.ref_use_rowid else 'False')}")
    lines.append(f"_ref_col = {repr(cfg.ref_col) if cfg.ref_col else 'None'}")
    lines.append("")

    # Build left key series
    lines.append("# Build key series for data (left) and reference (right)")
    lines.append("if _data_use_rowid:")
    lines.append("    _left_key = df_left.index.astype('string')")
    lines.append("else:")
    lines.append("    if _data_col is None or _data_col not in df_left.columns:")
    lines.append("        raise KeyError(f\"Reference Row Splitter: data key column not found: {_data_col!r}\")")
    lines.append("    _left_key = df_left[_data_col].astype('string')")
    lines.append("")
    # Build reference key set
    lines.append("if _ref_use_rowid:")
    lines.append("    _ref_key = df_right.index.astype('string')")
    lines.append("else:")
    lines.append("    if _ref_col is None or _ref_col not in df_right.columns:")
    lines.append("        raise KeyError(f\"Reference Row Splitter: reference key column not found: {_ref_col!r}\")")
    lines.append("    _ref_key = df_right[_ref_col].astype('string')")
    lines.append("")
    lines.append("# Membership mask (ignore NA in reference)")
    lines.append("_ref_set = set(_ref_key.dropna().tolist())")
    lines.append("_mask = _left_key.isin(_ref_set)")
    lines.append("")
    lines.append("# Outputs: port 1 = matching, port 2 = non-matching")
    lines.append("out_in  = df_left[_mask].copy()")
    lines.append("out_out = df_left[~_mask].copy()")
    lines.append(f"context['{node_id}:1'] = out_in")
    lines.append(f"context['{node_id}:2'] = out_out")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],        # two inputs: data, reference
    out_ports: Optional[List[str]] = None,  # ignored; fixed [1,2]
) -> List[str]:
    """
    Generate the Python body code for the node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports (ignored).

    Returns:
        List[str]: The lines of Python code for the node.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_refsplit_settings(ndir)

    # Determine inputs. Prefer (target_port → data/ref) if available; otherwise preserve order.
    # Here we receive normalized (src_id, src_port) pairs.
    pairs = normalize_in_ports(in_ports)
    left_src, left_in = pairs[0] if pairs else ("UNKNOWN", "1")
    right_src, right_in = pairs[1] if len(pairs) > 1 else ("UNKNOWN", "1")

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"df_left  = context['{left_src}:{left_in}']   # data table")
    lines.append(f"df_right = context['{right_src}:{right_in}']  # reference table")

    lines.extend(_emit_split_code(cfg, node_id))
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
        out_ports (Optional[List[str]]): The outgoing ports (ignored).

    Returns:
        str: The generated code as a string.
    """
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


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
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines.
    """
    explicit_imports = collect_module_imports(generate_imports)

    # Prefer target-port mapping when available
    data_pair: Optional[Tuple[str, str]] = None
    ref_pair: Optional[Tuple[str, str]] = None
    for src_id, e in (incoming or []):
        src_port = str(getattr(e, "source_port", "") or "1")
        tgt_port = str(getattr(e, "target_port", "") or "")
        if tgt_port == "1":
            data_pair = (str(src_id), src_port)
        elif tgt_port == "2":
            ref_pair = (str(src_id), src_port)

    norm_in: List[Tuple[str, str]] = []
    if data_pair:
        norm_in.append(data_pair)
    if ref_pair:
        norm_in.append(ref_pair)
    if not norm_in:
        # Fallback to the provided order
        norm_in = [(str(src), str(getattr(e, "source_port", "") or "1")) for src, e in (incoming or [])]

    node_lines = generate_py_body(nid, npath, norm_in)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
