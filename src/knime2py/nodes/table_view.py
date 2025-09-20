#!/usr/bin/env python3

####################################################################################################
#
# Table View
#
# Reads a single input table and prints it to stdout according to view settings in settings.xml:
# - Column selection: MANUAL list (selected_Internals + manualFilter/manuallySelected), with
#   includeUnknownColumns deciding whether to append other columns after the selection.
# - Banners: title, table size, and (optionally) column dtypes.
# - Display options: showRowIndices (controls index printing), simple pagination (first page only)
#   when enablePagination=true, honoring pageSize.
#
# This view node intentionally writes NO outputs to the workflow context – it only prints.
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory id
FACTORY = "org.knime.base.views.node.tableview.TableViewNodeFactory"

# --------------------------------------------------------------------------------------------------
# settings.xml → TableViewSettings
# --------------------------------------------------------------------------------------------------

@dataclass
class TableViewSettings:
    mode: str = "AUTO"                         # "MANUAL" | "AUTO"
    selected_cols: List[str] = field(default_factory=list)
    include_unknown: bool = False
    show_row_indices: bool = False
    show_row_keys: bool = True                 # (not used in this simple print view)
    title: str = "Table View"
    show_size: bool = True
    show_dtypes: bool = True
    enable_pagination: bool = False
    page_size: int = 10

def _bool(s: Optional[str], default: bool) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}

def _collect_numeric_name_entries(cfg: ET._Element) -> List[str]:
    out: List[str] = []
    for k, v in iter_entries(cfg):
        if k.isdigit() and v:
            out.append(v.strip())
    return out

def parse_table_view_settings(node_dir: Optional[Path]) -> TableViewSettings:
    if not node_dir:
        return TableViewSettings()

    sp = node_dir / "settings.xml"
    if not sp.exists():
        return TableViewSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    view = first_el(root, ".//*[local-name()='config' and @key='view']")

    if view is None:
        return TableViewSettings()

    # Displayed columns block
    disp = first_el(view, ".//*[local-name()='config' and @key='displayedColumns']")
    mode = (first(disp, ".//*[local-name()='entry' and @key='mode']/@value") or "AUTO").strip().upper() if disp is not None else "AUTO"

    selected: List[str] = []
    include_unknown = False

    if disp is not None:
        # 1) selected_Internals (current selection saved by UI)
        sel_int = first_el(disp, ".//*[local-name()='config' and @key='selected_Internals']")
        if sel_int is not None:
            selected.extend(_collect_numeric_name_entries(sel_int))
        # 2) manualFilter/manuallySelected (explicit manual list)
        man_sel = first_el(disp, ".//*[local-name()='config' and @key='manualFilter']/*[local-name()='config' and @key='manuallySelected']")
        if man_sel is not None:
            selected.extend(_collect_numeric_name_entries(man_sel))
        # includeUnknownColumns flag
        include_unknown = _bool(first(disp, ".//*[local-name()='config' and @key='manualFilter']/*[local-name()='entry' and @key='includeUnknownColumns']/@value"), False)

    # De-duplicate while preserving order
    selected = list(dict.fromkeys([c for c in selected if c]))

    # Other view flags
    show_row_indices = _bool(first(view, ".//*[local-name()='entry' and @key='showRowIndices']/@value"), False)
    show_row_keys    = _bool(first(view, ".//*[local-name()='entry' and @key='showRowKeys']/@value"), True)
    title            = first(view, ".//*[local-name()='entry' and @key='title']/@value") or "Table View"
    show_size        = _bool(first(view, ".//*[local-name()='entry' and @key='showTableSize']/@value"), True)
    show_dtypes      = _bool(first(view, ".//*[local-name()='entry' and @key='showColumnDataType']/@value"), True)
    enable_pagination = _bool(first(view, ".//*[local-name()='entry' and @key='enablePagination']/@value"), False)
    try:
        page_size = int(first(view, ".//*[local-name()='entry' and @key='pageSize']/@value") or 10)
    except Exception:
        page_size = 10

    return TableViewSettings(
        mode=mode,
        selected_cols=selected,
        include_unknown=include_unknown,
        show_row_indices=show_row_indices,
        show_row_keys=show_row_keys,
        title=title,
        show_size=show_size,
        show_dtypes=show_dtypes,
        enable_pagination=enable_pagination,
        page_size=page_size,
    )

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.views.node.tableview.TableViewNodeFactory"
)

def _emit_view_code(cfg: TableViewSettings) -> List[str]:
    lines: List[str] = []
    lines.append(f"_mode = {repr(cfg.mode)}")
    lines.append(f"_sel  = {repr(cfg.selected_cols)}")
    lines.append(f"_include_unknown = {repr(bool(cfg.include_unknown))}")
    lines.append(f"_title = {repr(cfg.title)}")
    lines.append(f"_show_size = {repr(bool(cfg.show_size))}")
    lines.append(f"_show_dtypes = {repr(bool(cfg.show_dtypes))}")
    lines.append(f"_show_index = {repr(bool(cfg.show_row_indices))}")
    lines.append(f"_pagination = {repr(bool(cfg.enable_pagination))}")
    lines.append(f"_page_size = int({int(cfg.page_size)})")
    lines.append("")
    lines.append("# Resolve columns to display")
    lines.append("if _mode == 'MANUAL' and _sel:")
    lines.append("    cols = [c for c in _sel if c in df.columns]")
    lines.append("    if _include_unknown:")
    lines.append("        cols = cols + [c for c in df.columns if c not in cols]")
    lines.append("else:")
    lines.append("    cols = list(df.columns)")
    lines.append("")
    lines.append("view_df = df[cols].copy() if cols else df.copy()")
    lines.append("")
    lines.append("# Banners")
    lines.append("print(f\"\\n=== {_title} ===\")")
    lines.append("if _show_size:")
    lines.append("    print(f\"[size] {view_df.shape[0]} rows × {view_df.shape[1]} columns\")")
    lines.append("if _show_dtypes and len(view_df.columns) > 0:")
    lines.append("    dts = view_df.dtypes.astype(str)")
    lines.append("    print('[dtypes] ' + ', '.join([f\"{c}: {dts[c]}\" for c in view_df.columns]))")
    lines.append("")
    lines.append("# Page selection (simple first-page preview when enabled)")
    lines.append("to_show = view_df")
    lines.append("if _pagination and _page_size > 0:")
    lines.append("    to_show = view_df.head(_page_size)")
    lines.append("")
    lines.append("# Print the table (respect showRowIndices)")
    lines.append("try:")
    lines.append("    if _show_index:")
    lines.append("        print(to_show.to_string(index=True))")
    lines.append("    else:")
    lines.append("        print(to_show.to_string(index=False))")
    lines.append("except Exception:")
    lines.append("    # Fallback: default print if to_string fails for any reason")
    lines.append("    print(to_show)")
    return lines

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,  # ignored
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_table_view_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Single table input (view-only; no outputs)
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    # Emit the print logic
    lines.extend(_emit_view_code(cfg))

    # No context outputs
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
    Returns (imports, body_lines) for this view node.
    """
    explicit_imports = collect_module_imports(generate_imports)

    # One upstream table expected
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    if not in_ports:
        # Still return a small stub that raises a clear error at runtime
        in_ports = [("UNKNOWN", "1")]

    node_lines = generate_py_body(nid, npath, in_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
