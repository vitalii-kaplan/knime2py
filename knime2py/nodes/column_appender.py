#!/usr/bin/env python3

####################################################################################################
#
# Column Appender
#
# Appends columns from one or more “right” tables to a single “left” table. Options are parsed from
# settings.xml. When row IDs are IDENTICAL the join aligns by index; otherwise it falls back to a
# positional column-wise concat. Right-hand name collisions are resolved by suffixing each right
# table’s columns with an incremented suffix (e.g., "_r1", "_r2", …).
#
# - Settings read: selected_rowid_mode, selected_rowid_table, selected_rowid_table_number
#   (base suffix defaults to "_r"; final suffix becomes f"{base}{k}" per right table).
# - Alignment: IDENTICAL → index join; other modes → positional concat with reset index.
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

# ---------------------------------------------------------------------
# settings.xml → ColumnAppenderSettings
# ---------------------------------------------------------------------

@dataclass
class ColumnAppenderSettings:
    rowid_mode: str = "IDENTICAL"   # KNIME "selected_rowid_mode"
    rowid_table: Optional[int] = None
    rowid_table_number: Optional[int] = None
    right_suffix: str = "_r"        # base suffix; final suffix becomes f"{right_suffix}{k}"


def _to_int_or_none(s: Optional[str]) -> Optional[int]:
    try:
        return int(s) if s is not None else None
    except Exception:
        return None


def parse_column_appender_settings(node_dir: Optional[Path]) -> ColumnAppenderSettings:
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


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    return ["import pandas as pd"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.columnappend2.ColumnAppender2NodeFactory"
)


def _emit_append_many_code(cfg: ColumnAppenderSettings, right_count: int) -> List[str]:
    """
    Return python lines that create out_df by appending columns of each right_df[k] to out_df.
    - Try index (row ID) alignment for IDENTICAL mode.
    - Rename right-side collisions with a per-right suffix (e.g., '_r1', '_r2', ...).
    - Fallback to positional concat when mode unsupported.
    """
    lines: List[str] = []
    lines.append("out_df = left_df.copy()")
    lines.append(f"_mode = {repr((cfg.rowid_mode or 'IDENTICAL').upper())}")
    lines.append(f"_base_suffix = {repr(cfg.right_suffix)}")
    lines.append("")
    lines.append("# Append each right table in order")
    lines.append("for k, rdf in enumerate(right_dfs, start=1):")
    lines.append("    # Collision-safe rename of this right-hand table")
    lines.append("    suff = f\"{_base_suffix}{k}\"")
    lines.append("    out_cols = set(out_df.columns)")
    lines.append("    collision_map = {c: f\"{c}{suff}\" for c in rdf.columns if c in out_cols}")
    lines.append("    right_safe = rdf.rename(columns=collision_map) if collision_map else rdf")
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
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_column_appender_settings(ndir)

    # Determine ordered inputs: prefer KNIME target_port to decide left vs rights
    # in_ports here is [(src_id, EdgeObj), ...] to allow target_port access
    # Build a list of (tgt_port_int_or_large, src_id, src_port)
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
            lines.append(f"right_dfs.append(context['{r_src}:{r_in}'])  # right table")
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
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Returns (imports, body_lines) if this module can handle the node; None otherwise.

    Port mapping:
      - Input 1 → left table
      - Input 2..N → right tables to append
      - Output 1 → appended table
    """

    explicit_imports = collect_module_imports(generate_imports)

    # Preserve (src_id, Edge) to keep target_port info for ordering
    in_ports = [(str(src_id), e) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
