#!/usr/bin/env python3

####################################################################################################
#
# Color Manager
#
# Passthrough implementation. 
# 
# KNIME color annotations are UI metadata and have no native representation in pandas;
# we therefore forward the input table unchanged to all outputs.
#
####################################################################################################

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

# Utilities from our framework
from .node_utils import (
    normalize_in_ports,
    collect_module_imports,
    split_out_imports,
    context_assignment_lines,
)

# KNIME factory for Color Manager (current)
FACTORY = "org.knime.base.node.viz.property.color.ColorManager2NodeFactory"

# Hub reference
HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.viz.property.color.ColorManager2NodeFactory"
)

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

def generate_imports():
    """
    Generate a list of imports required for the node.

    Returns:
        List: An empty list since no extra libraries are needed for passthrough.
    """
    return []

# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python code body for the node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The list of incoming ports.
        out_ports (Optional[List[str]]): The list of outgoing ports.

    Returns:
        List[str]: A list of strings representing the Python code body.
    """
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]  # first input
    ports = [str(p or "1") for p in (out_ports or ["1"])] or ["1"]

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append("# Color metadata is not preserved in pandas; passthrough the table unchanged.")
    lines.append(f"df = context['{src_id}:{in_port}']")
    # publish to all requested outputs
    for assign in context_assignment_lines(node_id, ports):
        lines.append(assign)
    return lines


def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    """
    Generate the code for the Jupyter notebook cell.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The list of incoming ports.
        out_ports (Optional[List[str]]): The list of outgoing ports.

    Returns:
        str: The generated code as a string.
    """
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"

# ---------------------------------------------------------------------
# Registry hook
# ---------------------------------------------------------------------

def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handle the node processing and return the necessary imports and body lines.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming connections.
        outgoing: The outgoing connections.

    Returns:
        Tuple: A tuple containing the imports and body lines.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
