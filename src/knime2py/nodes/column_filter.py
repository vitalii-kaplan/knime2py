#!/usr/bin/env python3

####################################################################################################
#
# Column Filter  (exclude-only)
#
# This exporter intentionally ignores any "include" lists in KNIME settings.xml and only applies
# the "exclude" list. The generated code simply drops those columns (if present) from the input df,
# preserving all remaining columns and their order.
#
# - Excludes are parsed heuristically from settings.xml by scanning <config> blocks whose keys
#   contain "exclude", collecting list entries (<entry key='0' value='Col'/> or <entry key='name'/>).
# - Dropping uses errors='ignore' so missing columns won't fail the cell.
# - If no excludes are found, the node is a passthrough.
#
# Depends on: lxml for parsing; pandas at runtime.
#
####################################################################################################

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
    return list(dict.fromkeys([s for s in seq if s]))


def parse_column_filter_settings(node_dir: Optional[Path]) -> ColumnFilterSettings:
    """
    Heuristic parser that extracts only the EXCLUDE column names from settings.xml.
    We look for <config> blocks whose @key contains 'exclude' and collect entries:
      - <entry key='0' value='Col'/> style lists
      - <entry key='name' value='Col'/> style lists
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


def generate_imports():
    return ["import pandas as pd"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    - Reads df from the first input port.
    - Drops exclude columns (if any).
    - Publishes to this node's output port(s).
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
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Returns (imports, body_lines).
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
