#!/usr/bin/env python3

####################################################################################################
#
# Concatenate
#
# Row-binds multiple input tables in port order and publishes the result to port 1.
#
# - No suffixing or renaming of columns.
# - No column intersection logic; pandas default union alignment is used.
# - Row index is reset (0..N-1) via ignore_index=True.
#
####################################################################################################

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .node_utils import (  # project helpers
    normalize_in_ports,
    collect_module_imports,
    split_out_imports,
)

# KNIME factory
FACTORY = "org.knime.base.node.preproc.append.row.AppendedRowsNodeFactory"

# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------

def generate_imports():
    """Generate a list of imports required for the concatenation operation."""
    return [
        "import pandas as pd",
    ]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.append.row.AppendedRowsNodeFactory"
)

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python code body for the concatenation operation.

    Args:
        node_id (str): The identifier for the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The list of incoming ports.
        out_ports (Optional[List[str]]): The list of outgoing ports.

    Returns:
        List[str]: The generated lines of Python code.
    """
    # Order inputs by target port index when available (1,2,3,...)
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
        # fallback to original order if registry didn’t pass target ports
        pairs = normalize_in_ports(in_ports)
        ordered = [(i + 1, sid, sp) for i, (sid, sp) in enumerate(pairs)]

    ordered.sort(key=lambda t: (t[0], t[1], t[2]))

    ports = [str(p or "1") for p in (out_ports or ["1"])]
    ports = list(dict.fromkeys(ports)) or ["1"]

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Gather inputs
    if ordered:
        lines.append("dfs = []")
        for _, sid, sp in ordered:
            lines.append(f"dfs.append(context['{sid}:{sp}'])")
    else:
        lines.append("dfs = []  # no inputs")

    # Simple row-wise concatenation; union alignment; reset index
    lines.append("out_df = (pd.concat(dfs, axis=0, ignore_index=True, sort=False) if dfs else pd.DataFrame())")

    # Publish single output (or mirror to multiple if requested)
    for p in ports:
        lines.append(f"context['{node_id}:{p}'] = out_df")

    return lines


def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    """
    Generate the code for the Jupyter notebook representation of the node.

    Args:
        node_id (str): The identifier for the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The list of incoming ports.
        out_ports (Optional[List[str]]): The list of outgoing ports.

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
        nid: The identifier for the node.
        npath: The path of the node.
        incoming: The incoming edges.
        outgoing: The outgoing edges.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines.
    """
    explicit_imports = collect_module_imports(generate_imports)

    # Preserve (src_id, Edge) so we can read target_port for ordering
    in_ports = [(str(src_id), e) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
