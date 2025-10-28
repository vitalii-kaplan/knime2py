#!/usr/bin/env python3

####################################################################################################
#
# Column Renamer
#
# Reads the renaming pairs from settings.xml and emits code that:
#   • Loads the single input table.
#   • Applies batch column renames using the configured (oldName → newName) pairs.
#   • Skips mappings where oldName is not present.
#   • Avoids target-name collisions by auto-suffixing ("_1", "_2", …) when needed.
#   • Writes the renamed table to the first (only) output port.
#
# • Supports only explicit (old → new) mappings from settings.xml.
# • No pattern/regex templating, no type-based renames, no column reordering.
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass, field
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
    iter_entries,
)

# KNIME factory
FACTORY = "org.knime.base.node.preproc.column.renamer.ColumnRenamerNodeFactory"

# --------------------------------------------------------------------------------------------------
# settings.xml → ColumnRenamerSettings
# --------------------------------------------------------------------------------------------------

@dataclass
class ColumnRenamerSettings:
    renamings: List[Tuple[str, str]] = field(default_factory=list)  # (old, new)

def _collect_renamings(cfg: ET._Element) -> List[Tuple[str, str]]:
    """
    Collects renaming pairs from the provided XML configuration element.

    Under <config key="renamings"> there are numbered <config key="N"> elements,
    each with entries: oldName, newName.

    Args:
        cfg (ET._Element): The XML configuration element.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing old and new column names.
    """
    out: List[Tuple[str, str]] = []
    if cfg is None:
        return out
    # children are <config key="0">, <config key="1">, ...
    for child in cfg:
        if not isinstance(child.tag, str):
            continue
        if child.tag.split('}')[-1] != 'config':
            continue
        # only numeric keys
        k = child.get('key')
        if k is None or not str(k).isdigit():
            continue
        old_name = first(child, ".//*[local-name()='entry' and @key='oldName']/@value")
        new_name = first(child, ".//*[local-name()='entry' and @key='newName']/@value")
        if old_name and new_name:
            out.append((old_name, new_name))
    return out

def parse_col_renamer_settings(node_dir: Optional[Path]) -> ColumnRenamerSettings:
    """
    Parses the column renamer settings from the settings.xml file.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        ColumnRenamerSettings: An instance containing the renaming pairs.
    """
    if not node_dir:
        return ColumnRenamerSettings()
    sp = node_dir / "settings.xml"
    if not sp.exists():
        return ColumnRenamerSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return ColumnRenamerSettings()

    ren_cfg = first_el(model, ".//*[local-name()='config' and @key='renamings']")
    pairs = _collect_renamings(ren_cfg)
    return ColumnRenamerSettings(renamings=pairs)

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    """
    Generates the necessary import statements for the code.

    Returns:
        List[str]: A list of import statements.
    """
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.column.renamer.ColumnRenamerNodeFactory"
)

def _emit_rename_code(cfg: ColumnRenamerSettings) -> List[str]:
    """
    Emits the code lines that apply the column renaming logic.

    This function builds a collision-safe mapping and applies DataFrame.rename().
    If a target name already exists (and is not the same column being renamed),
    it appends _1, _2, … until a unique name is found.

    Args:
        cfg (ColumnRenamerSettings): The settings containing the renaming pairs.

    Returns:
        List[str]: A list of code lines for renaming columns.
    """
    lines: List[str] = []
    # Serialize pairs for transparency
    if cfg.renamings:
        pairs_src = ", ".join(f"({repr(o)}, {repr(n)})" for (o, n) in cfg.renamings)
        lines.append(f"_pairs = [{pairs_src}]")
    else:
        lines.append("_pairs = []")

    lines += [
        "out_df = df.copy()",
        "# Filter to pairs where the source column exists",
        "present = [(o, n) for (o, n) in _pairs if o in out_df.columns]",
        "if not present:",
        "    final_map = {}",
        "else:",
        "    # Build a collision-safe mapping",
        "    existing = set(out_df.columns)",
        "    final_map = {}",
        "    for old, new in present:",
        "        # If renaming to the same name, keep it (no-op)",
        "        tgt = new",
        "        if tgt != old:",
        "            if tgt in existing and tgt != old:",
        "                base = tgt",
        "                k = 1",
        "                while f\"{base}_{k}\" in existing:",
        "                    k += 1",
        "                tgt = f\"{base}_{k}\"",
        "            # update sets for subsequent checks",
        "            existing.discard(old)",
        "            existing.add(tgt)",
        "        final_map[old] = tgt",
        "",
        "# Apply rename if any",
        "if final_map:",
        "    out_df = out_df.rename(columns=final_map)",
    ]
    return lines

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generates the Python code body for the column renamer node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The list of incoming ports.
        out_ports (Optional[List[str]]): The list of outgoing ports.

    Returns:
        List[str]: A list of code lines for the node's functionality.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_col_renamer_settings(ndir)

    # one input table
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]  # (source_node_id, source_port)
    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")
    lines.extend(_emit_rename_code(cfg))

    # single output
    ports = [str(p or "1") for p in (out_ports or ["1"])]
    ports = list(dict.fromkeys(ports)) or ["1"]
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
    Generates the code for a Jupyter notebook cell.

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

def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handles the processing of the node and returns the necessary imports and body lines.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming connections.
        outgoing: The outgoing connections.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines if the node can be handled; None otherwise.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
