#!/usr/bin/env python3

"""
Column Appender module.

Overview
----------------------------
This module generates Python code to append columns from one or more right DataFrames 
to a single left DataFrame, based on settings defined in settings.xml. It fits into 
the knime2py generator pipeline by transforming KNIME node configurations into executable 
Python code.

Runtime Behavior
----------------------------
Inputs:
- Reads DataFrames from the context using keys formatted as '{source_id}:{input_port}'.

Outputs:
- Writes the resulting DataFrame to the context under the key '{node_id}:{output_port}', 
  where output_port defaults to '1'.

Key algorithms:
- Implements collision handling for column names using KNIME-style suffixes to ensure 
  uniqueness when appending columns.

Edge Cases
----------------------------
The code handles scenarios such as empty or constant columns, NaNs, and ensures that 
the output DataFrame maintains integrity even when input DataFrames have mismatched 
indices or column names.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries: pandas. These dependencies 
are required for the generated code, not for this module itself.

Usage
----------------------------
Typically, this module is invoked by the knime2py emitter when processing a Column Appender 
node. An example of expected context access is:
```
left_df = context['source_id:1']
```

Node Identity
----------------------------
KNIME factory id:
- FACTORY = "org.knime.base.node.preproc.columnappend2.ColumnAppender2NodeFactory"

Configuration
----------------------------
The settings are encapsulated in the `ColumnAppenderSettings` dataclass, which includes:
- rowid_mode: Determines how to align rows (default is "IDENTICAL").
- rowid_table: Specifies the table to use for row IDs (default is None).
- rowid_table_number: Indicates the specific row ID table number (default is None).
The `parse_column_appender_settings` function extracts these values from settings.xml using 
XPath queries, with fallbacks to defaults.

Limitations
----------------------------
This module does not support all possible configurations available in KNIME and may 
approximate behavior in certain cases.

References
----------------------------
For more information, refer to the KNIME documentation and the following hub URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/org.knime.base.node.preproc.columnappend2.ColumnAppender2NodeFactory
"""

####################################################################################################
#
# Column Appender
#
# Appends columns from one or more “right” tables to a single “left” table. Options are parsed from
# settings.xml. When row IDs are IDENTICAL the join aligns by index; otherwise it falls back to a
# positional column-wise concat. Right-hand name collisions are resolved using KNIME-style per-right
# suffixes:  " (<#k>)"  → e.g.,  N_Voice (#1), Cardmon (#2), …
#
# - Settings read: selected_rowid_mode, selected_rowid_table, selected_rowid_table_number
# - Alignment: IDENTICAL → index join (reindex right to left if needed); otherwise positional concat
# - Suffixing rules:
#       • For each right table k (1-based), only columns that collide with *current* out_df get
#         renamed by appending " (#k)".
#       • If the new name would still collide (e.g., existing "A (#1)" already present), the suffix
#         is appended repeatedly: "A (#1) (#1)" and so on, until unique.
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (  # project helpers
    first,
    first_el,
    normalize_in_ports,
    collect_module_imports,
    split_out_imports,
)

# KNIME factory (Column Appender 2)
FACTORY = "org.knime.base.node.preproc.columnappend2.ColumnAppender2NodeFactory"

# --------------------------------------------------------------------------------------------------
# settings.xml → ColumnAppenderSettings
# --------------------------------------------------------------------------------------------------

@dataclass
class ColumnAppenderSettings:
    rowid_mode: str = "IDENTICAL"   # KNIME "selected_rowid_mode"
    rowid_table: Optional[int] = None
    rowid_table_number: Optional[int] = None


def _to_int_or_none(s: Optional[str]) -> Optional[int]:
    """
    Convert a string to an integer or return None if conversion fails.

    Args:
        s (Optional[str]): The string to convert.

    Returns:
        Optional[int]: The converted integer or None.
    """
    try:
        return int(s) if s is not None else None
    except Exception:
        return None


def parse_column_appender_settings(node_dir: Optional[Path]) -> ColumnAppenderSettings:
    """
    Parse the settings.xml file to extract ColumnAppenderSettings.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        ColumnAppenderSettings: The parsed settings.
    """
    if not node_dir:
        return ColumnAppenderSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return ColumnAppenderSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return ColumnAppenderSettings()

    mode = (first(model, ".//*[local-name()='entry' and @key='selected_rowid_mode']/@value") or "IDENTICAL").strip()
    tbl  = _to_int_or_none(first(model, ".//*[local-name()='entry' and @key='selected_rowid_table']/@value"))
    num  = _to_int_or_none(first(model, ".//*[local-name()='entry' and @key='selected_rowid_table_number']/@value"))

    return ColumnAppenderSettings(
        rowid_mode=mode or "IDENTICAL",
        rowid_table=tbl,
        rowid_table_number=num,
    )

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    """
    Generate the necessary imports for the Python code.

    Returns:
        List[str]: A list of import statements.
    """
    return ["import pandas as pd"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.columnappend2.ColumnAppender2NodeFactory"
)

def _emit_append_many_code(cfg: ColumnAppenderSettings, right_count: int) -> List[str]:
    """
    Generate Python code lines for appending columns from multiple right DataFrames to a left DataFrame.

    Args:
        cfg (ColumnAppenderSettings): The settings for the column appender.
        right_count (int): The number of right DataFrames.

    Returns:
        List[str]: The generated Python code lines.
    """
    lines: List[str] = []
    lines.append("out_df = left_df.copy()")
    lines.append(f"_mode = {repr((cfg.rowid_mode or 'IDENTICAL').upper())}")
    lines.append("")

    # Helper for KNIME-style unique names per right table
    lines.append("def _unique_with_suffix(base: str, existing: set, suffix: str) -> str:")
    lines.append("    if base not in existing:")
    lines.append("        return base")
    lines.append("    new = f\"{base}{suffix}\"")
    lines.append("    # If even the suffixed name exists, keep appending the same suffix")
    lines.append("    while new in existing:")
    lines.append("        new = f\"{new}{suffix}\"")
    lines.append("    return new")
    lines.append("")

    lines.append("# Append each right table in order (1-based index for suffix)")
    lines.append("for k, rdf in enumerate(right_dfs, start=1):")
    lines.append("    suffix = f\" (#{k})\"")
    lines.append("    # Build a collision-aware rename map guaranteeing uniqueness")
    lines.append("    existing = set(out_df.columns)")
    lines.append("    rename_map = {}")
    lines.append("    for c in rdf.columns:")
    lines.append("        if c in existing or c in rename_map.values():")
    lines.append("            newc = _unique_with_suffix(c, existing | set(rename_map.values()), suffix)")
    lines.append("            rename_map[c] = newc")
    lines.append("        else:")
    lines.append("            # no collision; keep original")
    lines.append("            rename_map[c] = c")
    lines.append("    right_safe = rdf.rename(columns=rename_map) if rename_map else rdf")
    lines.append("")
    lines.append("    if _mode == 'IDENTICAL':")
    lines.append("        # Align by index; if different, left-align by reindexing right to out_df.index")
    lines.append("        if out_df.index.equals(right_safe.index):")
    lines.append("            out_df = out_df.join(right_safe, how='left')")
    lines.append("        else:")
    lines.append("            out_df = out_df.join(right_safe.reindex(out_df.index), how='left')")
    lines.append("    else:")
    lines.append("        # Positional fallback for unknown modes")
    lines.append("        out_df = pd.concat([out_df.reset_index(drop=True),")
    lines.append("                           right_safe.reset_index(drop=True)], axis=1)")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],        # Port 1 = left, Ports 2..N = rights
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the body of the Python code for the node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The input ports.
        out_ports (Optional[List[str]]): The output ports.

    Returns:
        List[str]: The generated Python code lines.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_column_appender_settings(ndir)

    # Determine ordered inputs: prefer KNIME target_port to decide left vs rights
    ordered: List[Tuple[int, str, str]] = []
    for src_id, e in (in_ports or []):
        tgt = getattr(e, "target_port", None)
        try:
            tgt_i = int(tgt) if tgt is not None and str(tgt).strip().isdigit() else 999999
        except Exception:
            tgt_i = 999999
        src_port = str(getattr(e, "source_port", "") or "1")
        ordered.append((tgt_i, str(src_id), src_port))
    if not ordered:
        # fallback to simple normalize if no Edge objects with target_port
        pairs = normalize_in_ports(in_ports)
        ordered = [(i + 1, sid, sp) for i, (sid, sp) in enumerate(pairs)]

    ordered.sort(key=lambda t: (t[0], t[1], t[2]))

    # First becomes left; rest become rights
    left_tuple = ordered[0] if ordered else (1, "UNKNOWN", "1")
    right_tuples = ordered[1:] if len(ordered) > 1 else []

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Read inputs
    ltgt, l_src, l_in = left_tuple
    lines.append(f"left_df = context['{l_src}:{l_in}']  # left table")
    if right_tuples:
        lines.append("right_dfs = []")
        for _, r_src, r_in in right_tuples:
            lines.append(f"right_dfs.append(context['{r_src}:{r_in}'])")
    else:
        lines.append("right_dfs = []  # no additional inputs; passthrough")

    # Append logic
    lines.extend(_emit_append_many_code(cfg, right_count=len(right_tuples)))

    # Publish to context (default single output port '1')
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
        in_ports (List[object]): The input ports.
        out_ports (Optional[List[str]]): The output ports.

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
        incoming: The incoming edges.
        outgoing: The outgoing edges.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines.
    """
    explicit_imports = collect_module_imports(generate_imports)

    # Preserve (src_id, Edge) to keep target_port info for ordering
    in_ports = [(str(src_id), e) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
