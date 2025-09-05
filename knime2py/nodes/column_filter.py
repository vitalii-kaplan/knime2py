#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # re-use shared XML + port helpers, as agreed


# KNIME Base Column Filter factories seen across versions
COLUMN_FILTER_FACTORIES = "org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory"

def can_handle(node_type: Optional[str]) -> bool:
    if not node_type:
        return False
    return node_type.endswith(COLUMN_FILTER_FACTORIES)


# ---------------------------------------------------------------------
# settings.xml → ColumnFilterSettings
# ---------------------------------------------------------------------

@dataclass
class ColumnFilterSettings:
    includes: List[str] = field(default_factory=list)
    excludes: List[str] = field(default_factory=list)


def _gather_numeric_entries(cfg: ET._Element) -> List[str]:
    """
    Extract values from <entry key="0" value="..."/>, <entry key="1" ...> ... patterns
    within a subtree.
    """
    out: List[str] = []
    for ent in cfg.xpath(".//*[local-name()='entry']"):
        k = (ent.get("key") or "").strip()
        v = (ent.get("value") or "").strip()
        if k.isdigit() and v:
            out.append(v)
    return out


def _gather_named_entries(cfg: ET._Element) -> List[str]:
    """
    Extract values from <entry key='name' value='col'/> patterns somewhere below cfg.
    KNIME often nests columns like <config key='columns'><config key='0'><entry key='name' ...>
    """
    return [
        (v or "").strip()
        for v in cfg.xpath(".//*[local-name()='entry' and translate(@key,"
                           " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='name']/@value")
        if v
    ]


def _uniq_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def parse_column_filter_settings(node_dir: Optional[Path]) -> ColumnFilterSettings:
    """
    Heuristic parser that finds include/exclude column names in Column Filter settings.xml.
    Handles a few common KNIME layouts:
      - lists under any config whose key contains 'include'/'exclude'
      - numeric entry lists (<entry key='0' value='Col'/> ...)
      - nested column blocks with <entry key='name' value='Col'/>
    """
    if not node_dir:
        return ColumnFilterSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return ColumnFilterSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    # Collect INCLUDE candidates
    include_cfgs = root.xpath(
        ".//*[local-name()='config' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'include')]"
    )
    includes: List[str] = []
    for cfg in include_cfgs:
        includes.extend(_gather_numeric_entries(cfg))
        includes.extend(_gather_named_entries(cfg))

    # Collect EXCLUDE candidates
    exclude_cfgs = root.xpath(
        ".//*[local-name()='config' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'exclude')]"
    )
    excludes: List[str] = []
    for cfg in exclude_cfgs:
        excludes.extend(_gather_numeric_entries(cfg))
        excludes.extend(_gather_named_entries(cfg))

    # As a fallback, some exports put explicit columns under generic 'columns'
    # blocks with an 'include'/'exclude' switch at a sibling. We try a broad sweep:
    if not includes and not excludes:
        for cfg in root.xpath(".//*[local-name()='config' and contains(translate(@key,"
                              " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'columns')]"):
            names = _gather_named_entries(cfg) or _gather_numeric_entries(cfg)
            # If we can’t detect mode, default to includes (safe/explicit)
            includes.extend(names)

    return ColumnFilterSettings(
        includes=_uniq_preserve([c for c in includes if c]),
        excludes=_uniq_preserve([c for c in excludes if c]),
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

HUB_URL = "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/" \
          "org.knime.base.node.preproc.filter.column2.ColumnFilter2NodeFactory"

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
        # drop only those present; errors='ignore' is fine but we also filter for clarity
        lines.append("cols_exc = [c for c in exclude_cols if c in out_df.columns]")
        lines.append("out_df = out_df.drop(columns=cols_exc, errors='ignore')")

    if not (have_inc or have_exc):
        lines.append("# No explicit include/exclude columns found in settings.xml; passthrough.")
        lines.append("out_df = df")

    return lines


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
    lines.append("import pandas as pd  # required at runtime")
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
