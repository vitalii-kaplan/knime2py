#!/usr/bin/env python3

"""
Column Filter module for KNIME to Python conversion.

Overview
----------------------------
This module generates Python code that filters columns from a DataFrame based on
exclusion criteria specified in a KNIME node's settings.xml file. It fits into the
knime2py generator pipeline by transforming the configuration into executable Python
code.

Runtime Behavior
----------------------------
Inputs:
- Reads a DataFrame from the context using the key format 'src_id:in_port'.

Outputs:
- Writes the filtered DataFrame to the context using the key format 'node_id:out_port',
  where out_port defaults to '1'.

Key algorithms:
- The module implements a column exclusion mechanism, dropping specified columns while
  preserving the order of the remaining columns.

Edge Cases
----------------------------
The code handles cases where:
- No columns are specified for exclusion, resulting in a passthrough.
- Missing columns in the DataFrame do not raise errors due to the use of
  `errors='ignore'` in the drop operation.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries:
- pandas

These dependencies are required by the generated code, not by this module.

Usage
----------------------------
This module is typically invoked by the knime2py emitter when processing a Column Filter
node. An example of expected context access is:
```python
df = context['source_id:1']  # Accessing the input DataFrame
```

Node Identity
----------------------------
KNIME factory id:
- FACTORY = "org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory"

Configuration
----------------------------
The settings are encapsulated in the `ColumnFilterSettings` dataclass, which contains:
- excludes: List of column names to exclude (default is an empty list).

The `parse_column_filter_settings` function extracts these values by scanning the
settings.xml for configuration blocks containing 'exclude'.

Limitations
----------------------------
This module does not support inclusion lists; it only processes exclusion criteria.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.node.preproc.filter.column2.ColumnFilter2NodeFactory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory id (Column Filter / legacy)
FACTORY = "org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory"

# ---------------------------------------------------------------------
# settings.xml → ColumnFilterSettings
# ---------------------------------------------------------------------

@dataclass
class ColumnFilterSettings:
    excludes: List[str] = field(default_factory=list)  # includes intentionally ignored


def _uniq_preserve(seq: List[str]) -> List[str]:
    """
    Remove duplicates from a list while preserving the order of elements.

    Args:
        seq (List[str]): The input list from which to remove duplicates.

    Returns:
        List[str]: A list with duplicates removed, preserving the original order.
    """
    return list(dict.fromkeys([s for s in seq if s]))


def parse_column_filter_settings(node_dir: Optional[Path]) -> ColumnFilterSettings:
    """
    Heuristic parser that extracts only the EXCLUDE column names from settings.xml.
    We look for <config> blocks whose @key contains 'exclude' and collect entries:
      - <entry key='0' value='Col'/> style lists
      - <entry key='name' value='Col'/> style lists

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        ColumnFilterSettings: An instance of ColumnFilterSettings containing the excluded column names.
    """
    if not node_dir:
        return ColumnFilterSettings()
    sp = node_dir / "settings.xml"
    if not sp.exists():
        return ColumnFilterSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()

    def _collect_from_cfgs(cfgs) -> List[str]:
        out: List[str] = []
        for cfg in cfgs:
            for k, v in iter_entries(cfg):  # project helper
                if v is None:
                    continue
                lk = str(k).lower()
                if k.isdigit() or lk == "name":
                    out.append(v)
        return out

    exclude_cfgs = root.xpath(
        ".//*[local-name()='config' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'exclude')]"
    )
    excludes = _collect_from_cfgs(exclude_cfgs)

    return ColumnFilterSettings(excludes=_uniq_preserve(excludes))


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.filter.column2.ColumnFilter2NodeFactory"
)

def _emit_filter_code(settings: ColumnFilterSettings) -> List[str]:
    """
    Return python lines that transform `df` into `out_df` by DROPPING excludes only.

    Args:
        settings (ColumnFilterSettings): The settings containing the excluded column names.

    Returns:
        List[str]: A list of Python code lines for dropping the excluded columns.
    """
    lines: List[str] = []
    lines.append("out_df = df")
    if settings.excludes:
        exc_list = ", ".join(repr(c) for c in settings.excludes)
        lines.append(f"exclude_cols = [{exc_list}]")
        lines.append("out_df = out_df.drop(columns=[c for c in exclude_cols if c in out_df.columns], errors='ignore')")
    else:
        lines.append("# No excludes found; passthrough.")
    return lines


def generate_imports() -> List[str]:
    """
    Generate the necessary import statements for the generated Python code.

    Returns:
        List[str]: A list of import statements.
    """
    return ["import pandas as pd"]


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
        in_ports (List[object]): The input ports for the node.
        out_ports (Optional[List[str]]): The output ports for the node.

    Returns:
        List[str]: A list of Python code lines for the node's functionality.
    """
    ndir = Path(node_dir) if node_dir else None
    settings = parse_column_filter_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_filter_code(settings))

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
    Generate the complete code for a Jupyter notebook cell.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The input ports for the node.
        out_ports (Optional[List[str]]): The output ports for the node.

    Returns:
        str: The generated code as a string.
    """
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handle the processing of a node, returning the necessary imports and body lines.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path to the node.
        incoming: The incoming connections to the node.
        outgoing: The outgoing connections from the node.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the list of imports and the body lines.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
