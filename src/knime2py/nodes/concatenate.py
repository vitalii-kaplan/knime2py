#!/usr/bin/env python3

####################################################################################################
#
# Concatenate
#
# Row-binds multiple input tables. Options parsed from settings.xml:
#   • fail_on_duplicates: if True, raise if any row IDs (indexes) overlap across inputs.
#   • append_suffix:      when duplicate column names appear from later tables, rename them
#                         by appending <suffix>, and if still colliding, append a counter
#                         (<suffix>2, <suffix>3, …).
#   • intersection_of_columns: if True, keep only the intersection of columns across inputs.
#                              Otherwise take the union; missing values are filled with NaN.
#   • suffix: the text appended to resolve duplicate column names (default "_dup").
# Result is written to the first output port. 
# 
# Row index is reset to a simple 0..N-1 range.
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

# KNIME factory
FACTORY = "org.knime.base.node.preproc.append.row.AppendedRowsNodeFactory"

# --------------------------------------------------------------------------------------------------
# settings.xml → ConcatenateSettings
# --------------------------------------------------------------------------------------------------

@dataclass
class ConcatenateSettings:
    fail_on_duplicates: bool = False
    append_suffix: bool = True
    intersection_of_columns: bool = False
    suffix: str = "_dup"
    # hiliting flag exists but is irrelevant for codegen
    enable_hiliting: bool = False


def _to_bool(s: Optional[str], default: bool = False) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def parse_concatenate_settings(node_dir: Optional[Path]) -> ConcatenateSettings:
    if not node_dir:
        return ConcatenateSettings()
    sp = node_dir / "settings.xml"
    if not sp.exists():
        return ConcatenateSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return ConcatenateSettings()

    fail_dups = _to_bool(first(model, ".//*[local-name()='entry' and @key='fail_on_duplicates']/@value"), False)
    app_sfx   = _to_bool(first(model, ".//*[local-name()='entry' and @key='append_suffix']/@value"), True)
    inter     = _to_bool(first(model, ".//*[local-name()='entry' and @key='intersection_of_columns']/@value"), False)
    suffix    = first(model, ".//*[local-name()='entry' and @key='suffix']/@value") or "_dup"
    hilite    = _to_bool(first(model, ".//*[local-name()='entry' and @key='enable_hiliting']/@value"), False)

    return ConcatenateSettings(
        fail_on_duplicates=fail_dups,
        append_suffix=app_sfx,
        intersection_of_columns=inter,
        suffix=suffix,
        enable_hiliting=hilite,
    )

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    return [
        "import pandas as pd",
        "import numpy as np",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.append.row.AppendedRowsNodeFactory"
)


def _emit_concat_code(cfg: ConcatenateSettings) -> List[str]:
    """
    Emit robust row-wise concatenation with:
      - optional row-ID (index) duplicate checks,
      - duplicate column renaming for later tables,
      - union or intersection alignment of columns,
      - stable column order based on first appearance.
    """
    lines: List[str] = []
    lines.append(f"_fail_dups = {bool(cfg.fail_on_duplicates)}")
    lines.append(f"_append_suffix = {bool(cfg.append_suffix)}")
    lines.append(f"_suffix = {repr(cfg.suffix or '_dup')}")
    lines.append(f"_intersection = {bool(cfg.intersection_of_columns)}")
    lines.append("")
    lines.append("if not dfs:")
    lines.append("    out_df = pd.DataFrame()")
    lines.append("else:")
    lines.append("    # Prepare first table as the base")
    lines.append("    out_df = dfs[0].copy()")
    lines.append("    seen_cols = list(out_df.columns)")
    lines.append("    seen_set = set(seen_cols)")
    lines.append("    seen_index_sets = [set(out_df.index)]")
    lines.append("")
    lines.append("    # Helper: ensure unique column names for a right table against `seen_set`")
    lines.append("    def _rename_with_suffix(rdf, base_suffix):")
    lines.append("        if not _append_suffix:")
    lines.append("            return rdf")
    lines.append("        rename_map = {}")
    lines.append("        taken = set(seen_set)  # current global taken names")
    lines.append("        for c in rdf.columns:")
    lines.append("            if c in taken:")
    lines.append("                base = f\"{c}{base_suffix}\"")
    lines.append("                new_name = base")
    lines.append("                k = 2")
    lines.append("                while new_name in taken:")
    lines.append("                    new_name = f\"{base}{k}\"")
    lines.append("                    k += 1")
    lines.append("                rename_map[c] = new_name")
    lines.append("                taken.add(new_name)")
    lines.append("            else:")
    lines.append("                taken.add(c)")
    lines.append("        return rdf.rename(columns=rename_map) if rename_map else rdf")
    lines.append("")
    lines.append("    # Process remaining tables")
    lines.append("    for i, rdf0 in enumerate(dfs[1:], start=2):")
    lines.append("        rdf = _rename_with_suffix(rdf0, _suffix)")
    lines.append("")
    lines.append("        # Column alignment: intersection vs union")
    lines.append("        if _intersection:")
    lines.append("            common = [c for c in seen_cols if c in rdf.columns]")
    lines.append("            out_df = out_df[common]")
    lines.append("            rdf = rdf[common]")
    lines.append("            cols_order = common")
    lines.append("        else:")
    lines.append("            # Union with order: keep existing order, then add new columns as they appear")
    lines.append("            new_cols = [c for c in rdf.columns if c not in seen_set]")
    lines.append("            cols_order = seen_cols + new_cols")
    lines.append("            # Align both sides to the union")
    lines.append("            out_df = out_df.reindex(columns=cols_order)")
    lines.append("            rdf = rdf.reindex(columns=cols_order)")
    lines.append("            # Update seen lists/sets")
    lines.append("            for c in new_cols:")
    lines.append("                if c not in seen_set:")
    lines.append("                    seen_set.add(c)")
    lines.append("                    seen_cols.append(c)")
    lines.append("")
    lines.append("        # Optional: enforce distinct row IDs across inputs")
    lines.append("        if _fail_dups:")
    lines.append("            cur_idx = set(rdf.index)")
    lines.append("            for prev in seen_index_sets:")
    lines.append("                if prev & cur_idx:")
    lines.append("                    raise ValueError('Concatenate: duplicate row IDs detected across inputs; enable index reset or disable fail_on_duplicates in KNIME.')")
    lines.append("            seen_index_sets.append(cur_idx)")
    lines.append("")
    lines.append("        # Append rows; ignore original indices to match KNIME’s generated RowIDs")
    lines.append("        out_df = pd.concat([out_df, rdf], axis=0, ignore_index=True, sort=False)")
    lines.append("")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_concatenate_settings(ndir)

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
        pairs = normalize_in_ports(in_ports)
        ordered = [(i + 1, sid, sp) for i, (sid, sp) in enumerate(pairs)]
    ordered.sort(key=lambda t: (t[0], t[1], t[2]))

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Read all inputs in order
    if ordered:
        lines.append("dfs = []")
        for _, sid, sp in ordered:
            lines.append(f"dfs.append(context['{sid}:{sp}'])")
    else:
        lines.append("dfs = []  # no inputs")

    # Concatenate logic
    lines.extend(_emit_concat_code(cfg))

    # Publish (single output)
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
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Returns (imports, body_lines) if this module can handle the node; None otherwise.

    Ports:
      • Inputs 1..N → tables to be row-bound in port order
      • Output 1    → concatenated table
    """
    explicit_imports = collect_module_imports(generate_imports)

    # Preserve (src_id, Edge) to keep target_port for ordering
    in_ports = [(str(src_id), e) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
