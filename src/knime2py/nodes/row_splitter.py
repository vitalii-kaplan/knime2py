#!/usr/bin/env python3

"""Row Splitter node."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .node_utils import collect_module_imports, normalize_in_ports, split_out_imports
from .row_filter import RowFilterSettings, _emit_filter_code, parse_row_filter_settings


FACTORY = "org.knime.base.node.preproc.filter.row3.RowSplitterNodeFactory"


def generate_imports() -> List[str]:
    return ["import pandas as pd", "import re as _re"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.filter.row3.RowSplitterNodeFactory"
)


def _emit_split_code(cfg: RowFilterSettings) -> List[str]:
    lines = _emit_filter_code(cfg)
    lines.append("_matching_df = out_df")
    lines.append("_non_matching_df = df.loc[~final_mask].copy() if 'final_mask' in locals() else df.iloc[0:0].copy()")
    if cfg.output_mode.upper() == "NON_MATCHING":
        lines.append("_port1_df = _non_matching_df")
        lines.append("_port2_df = _matching_df")
    else:
        lines.append("_port1_df = _matching_df")
        lines.append("_port2_df = _non_matching_df")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_row_filter_settings(ndir)

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0] if pairs else ("UNKNOWN", "1")

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")
    lines.extend(_emit_split_code(cfg))

    port_map = {"1": "_port1_df", "2": "_port2_df"}
    for p in sorted({str(p or '1') for p in (out_ports or ['1', '2'])}):
        lines.append(f"context['{node_id}:{p}'] = {port_map.get(p, '_port1_df')}")
    return lines


def get_name() -> str:
    return "Row Splitter"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src_id, str(getattr(edge, "source_port", "") or "1")) for src_id, edge in (incoming or [])]
    out_ports = [str(getattr(edge, "source_port", "") or "1") for _, edge in (outgoing or [])] or ["1", "2"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
