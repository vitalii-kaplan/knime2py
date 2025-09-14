#!/usr/bin/env python3

####################################################################################################
#
# Column Filter
#
# Filters columns of the input table according to include/exclude lists parsed from KNIME
# settings.xml. The generated pandas code preserves include order, applies excludes, and writes
# the result to this node's context output port(s).
#
# Supports multiple KNIME factories:
#  - org.knime.base.node.preproc.filter.column2.ColumnFilter2NodeFactory (newer)
#  - org.knime.base.node.preproc.colfilter.ColumnFilterNodeFactory (classic)
#  - org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory (legacy/alt)
# Parsing heuristics: looks for <config> blocks whose keys contain "include"/"exclude", numeric
# index entries (<entry key='0' value='Col'/> ...), or name entries (<entry key='name' value='Col'/>).
# Falls back to generic "columns" blocks if include/exclude buckets are absent. Duplicate names are
# de-duplicated while preserving the first occurrence. If no includes/excludes are found, the node
# is a passthrough. Excludes are dropped with errors='ignore'. Only explicit column-name lists are
# supported—pattern/type/regex-based selection is not implemented. Depends on lxml for parsing and
# emits pandas-only code; relies on project utilities (iter_entries, normalize_in_ports, etc.).
#
####################################################################################################


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import * 


# Support multiple Column Filter factories across KNIME versions
COLUMN_FILTER_FACTORIES = (
    "org.knime.base.node.preproc.filter.column2.ColumnFilter2NodeFactory",   # newer
    "org.knime.base.node.preproc.colfilter.ColumnFilterNodeFactory",         # classic
    "org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory",  # legacy/alt
)

def can_handle(node_type: Optional[str]) -> bool:
    return bool(node_type and any(node_type.endswith(sfx) for sfx in COLUMN_FILTER_FACTORIES))


# ---------------------------------------------------------------------
# settings.xml → ColumnFilterSettings
# ---------------------------------------------------------------------

@dataclass
class ColumnFilterSettings:
    includes: List[str] = field(default_factory=list)
    excludes: List[str] = field(default_factory=list)


def _uniq_preserve(seq: List[str]) -> List[str]:
    # simple, fast order-preserving uniquifier
    return list(dict.fromkeys([s for s in seq if s]))


def parse_column_filter_settings(node_dir: Optional[Path]) -> ColumnFilterSettings:
    """
    Heuristic parser that finds include/exclude column names in Column Filter settings.xml.
    Handles common KNIME layouts:
      - any <config> whose key contains 'include'/'exclude'
      - numeric entry lists (<entry key='0' value='Col'/> ...)
      - nested blocks with <entry key='name' value='Col'/>
    """
    if not node_dir:
        return ColumnFilterSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return ColumnFilterSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    def _collect_from_cfgs(cfgs) -> List[str]:
        out: List[str] = []
        for cfg in cfgs:
            for k, v in iter_entries(cfg):  # from node_utils (regex-friendly)
                lk = k.lower()
                if (k.isdigit() or lk == "name") and v:
                    out.append(v)
        return out

    # INCLUDE / EXCLUDE via config key tokens
    include_cfgs = root.xpath(
        ".//*[local-name()='config' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'include')]"
    )
    exclude_cfgs = root.xpath(
        ".//*[local-name()='config' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'exclude')]"
    )

    includes = _collect_from_cfgs(include_cfgs)
    excludes = _collect_from_cfgs(exclude_cfgs)

    # Fallback: generic 'columns' blocks when include/exclude buckets aren’t explicit
    if not includes and not excludes:
        columns_cfgs = root.xpath(
            ".//*[local-name()='config' and contains(translate(@key,"
            " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'columns')]"
        )
        includes.extend(_collect_from_cfgs(columns_cfgs))

    return ColumnFilterSettings(
        includes=_uniq_preserve(includes),
        excludes=_uniq_preserve(excludes),
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.filter.column2.ColumnFilter2NodeFactory"
)

def _emit_filter_code(settings: ColumnFilterSettings) -> List[str]:
    """
    Return python lines that transform `df` into `out_df` using includes/excludes.
    We preserve order for includes; excludes are removed if present.
    """
    lines: List[str] = []

    have_inc = bool(settings.includes)
    have_exc = bool(settings.excludes)

    if have_inc:
        inc_list = ", ".join(repr(c) for c in settings.includes)
        lines.append(f"include_cols = [{inc_list}]")
        lines.append("cols_inc = [c for c in include_cols if c in df.columns]")
        lines.append("out_df = df[cols_inc]  # keep order")
    else:
        lines.append("out_df = df")

    if have_exc:
        exc_list = ", ".join(repr(c) for c in settings.excludes)
        lines.append(f"exclude_cols = [{exc_list}]")
        lines.append("cols_exc = [c for c in exclude_cols if c in out_df.columns]")
        lines.append("out_df = out_df.drop(columns=cols_exc, errors='ignore')")

    if not (have_inc or have_exc):
        lines.append("# No explicit include/exclude columns found in settings.xml; passthrough.")
        lines.append("out_df = df")

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
    Body lines for the .py workbook.
    - Reads df from the FIRST in_port (Column Filter has a single table input).
    - Applies includes/excludes.
    - Publishes to this node's context key(s).
    """
    ndir = Path(node_dir) if node_dir else None
    settings = parse_column_filter_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_filter_code(settings))

    # Publish to context (default to port '1' if not provided)
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
    """Single string for a notebook code cell."""
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    if not (ntype and can_handle(ntype)):
        return None

    # explicit imports declared by this node module
    explicit_imports = collect_module_imports(generate_imports)

    # ports
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    # single call with BOTH in/out ports
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)

    # split inline imports out of the body
    found_imports, body = split_out_imports(node_lines)

    # merge explicit + found
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body