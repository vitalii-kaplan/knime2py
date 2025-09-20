#!/usr/bin/env python3

####################################################################################################
#
# Normalizer
#
# Normalizes selected columns using Min–Max or Z-Score according to settings.xml, then writes the
# result to this node's context. Includes/excludes are honored; if no columns are selected the node
# is a passthrough.
#
# - Column selection: use included_names if set; else all numeric dtypes (Int*/int*/Float*/float*);
#   drop excluded_names afterward.
# - Modes: MINMAX uses new-min/new-max (constant/empty columns map to new_min);
#   ZSCORE uses (x-mean)/std (zero std → 0.0).
#
####################################################################################################


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import * 


# KNIME factory for Normalizer
FACTORY = "org.knime.base.node.preproc.normalize3.Normalizer3NodeFactory"

# ---------------------------------------------------------------------
# settings.xml → NormalizerSettings
# ---------------------------------------------------------------------

@dataclass
class NormalizerSettings:
    mode: str = "MINMAX"        # Supported: "MINMAX", "ZSCORE"
    new_min: float = 0.0        # only used for MINMAX
    new_max: float = 1.0        # only used for MINMAX
    includes: List[str] = field(default_factory=list)
    excludes: List[str] = field(default_factory=list)


def _collect_numeric_name_entries(cfg: ET._Element) -> List[str]:
    """Collect <entry key='0' value='col'>, <entry key='1' value='col'> … under cfg."""
    out: List[str] = []
    for k, v in iter_entries(cfg):
        if k.isdigit() and v:
            out.append(v.strip())
    return out


def parse_normalizer_settings(node_dir: Optional[Path]) -> NormalizerSettings:
    """
    Parse Normalizer settings:
      - mode: entry key='mode' (e.g. MINMAX, ZSCORE)
      - new-min/new-max (for MINMAX)
      - column filter: model/data-column-filter/{included_names, excluded_names}
    """
    if not node_dir:
        return NormalizerSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return NormalizerSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    # model element (get the actual element, not a string)
    model_cfgs = root.xpath(".//*[local-name()='config' and @key='model']")
    if not model_cfgs:
        return NormalizerSettings()
    model = model_cfgs[0]

    # Mode + params
    mode = (first(model, ".//*[local-name()='entry' and @key='mode']/@value") or "MINMAX").strip().upper()

    def _float_entry(key: str, default: float) -> float:
        v = first(model, f".//*[local-name()='entry' and @key='{key}']/@value")
        try:
            return float(v) if v is not None else default
        except Exception:
            return default

    new_min = _float_entry("new-min", 0.0)
    new_max = _float_entry("new-max", 1.0)

    # Column filter include/exclude (work with elements directly)
    includes: List[str] = []
    excludes: List[str] = []

    dcf_els = model.xpath(".//*[local-name()='config' and @key='data-column-filter']")
    if dcf_els:
        dcf_el = dcf_els[0]

        inc_cfgs = dcf_el.xpath(".//*[local-name()='config' and @key='included_names']")
        if inc_cfgs:
            includes.extend(_collect_numeric_name_entries(inc_cfgs[0]))

        exc_cfgs = dcf_el.xpath(".//*[local-name()='config' and @key='excluded_names']")
        if exc_cfgs:
            excludes.extend(_collect_numeric_name_entries(exc_cfgs[0]))

    # Uniquify, preserve order
    def _uniq_preserve(seq: List[str]) -> List[str]:
        return list(dict.fromkeys([s for s in seq if s]))

    return NormalizerSettings(
        mode=mode,
        new_min=new_min,
        new_max=new_max,
        includes=_uniq_preserve(includes),
        excludes=_uniq_preserve(excludes),
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    # Keep it minimal; numpy isn’t strictly required for the emitted code below
    return ["import pandas as pd"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.normalize3.Normalizer3NodeFactory"
)

_NUMERIC_DTYPES = "['Int64', 'Int32', 'Int16', 'int64', 'int32', 'int16', 'Float64', 'float64', 'float32']"


def _emit_normalize_code(cfg: NormalizerSettings) -> List[str]:
    """
    Emit lines that create `out_df` by normalizing a selected set of columns.
    Selection priority:
      1) If includes present -> start with those (existing in df)
      2) Else -> all numeric columns
      3) Then drop any excludes that are present
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")

    # Build column list
    if cfg.includes:
        inc_list = ", ".join(repr(c) for c in cfg.includes)
        lines.append(f"include_cols = [{inc_list}]")
        lines.append("norm_cols = [c for c in include_cols if c in out_df.columns]")
    else:
        lines.append(f"norm_cols = out_df.select_dtypes(include={_NUMERIC_DTYPES}).columns.tolist()")

    if cfg.excludes:
        exc_list = ", ".join(repr(c) for c in cfg.excludes)
        lines.append(f"exclude_cols = [{exc_list}]")
        lines.append("norm_cols = [c for c in norm_cols if c not in set(exclude_cols)]")

    # No columns? passthrough hint
    lines.append("if not norm_cols:")
    lines.append("    # Nothing to normalize; passthrough")
    lines.append("    pass")
    lines.append("else:")

    mode = (cfg.mode or "MINMAX").upper()

    if mode == "MINMAX":
        # Min-Max: new_min + (x - min) / (max - min) * (new_max - new_min), guard zero-range
        lines.append(f"    _new_min, _new_max = {cfg.new_min}, {cfg.new_max}")
        lines.append("    _span = (_new_max - _new_min)")
        lines.append("    def _minmax(s):")
        lines.append("        mn, mx = s.min(), s.max()")
        lines.append("        rng = mx - mn")
        lines.append("        if pd.isna(rng) or rng == 0:")
        lines.append("            # constant/empty column → map to new_min")
        lines.append("            return pd.Series([_new_min] * len(s), index=s.index)")
        lines.append("        return _new_min + (s - mn) / rng * _span")
        lines.append("    out_df[norm_cols] = out_df[norm_cols].apply(_minmax)")

    elif mode == "ZSCORE":
        # Standard score: (x - mean) / std, guard zero std
        lines.append("    def _zscore(s):")
        lines.append("        mu, sd = s.mean(), s.std()")
        lines.append("        if pd.isna(sd) or sd == 0:")
        lines.append("            return pd.Series([0.0] * len(s), index=s.index)")
        lines.append("        return (s - mu) / sd")
        lines.append("    out_df[norm_cols] = out_df[norm_cols].apply(_zscore)")

    else:
        # Unknown → leave a stub and passthrough
        lines.append(f"    # TODO: Unsupported Normalizer mode '{mode}'; leaving columns unchanged")

    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_normalizer_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_normalize_code(cfg))

    # Publish to context (default port '1')
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
    Central entry used by emitters:
      - returns (imports, body_lines) if this module can handle the node type
      - returns None otherwise
    """

    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
