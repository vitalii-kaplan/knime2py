#!/usr/bin/env python3

####################################################################################################
#
# One to Many (One-Hot Encoding)
#
# Transforms selected nominal/string columns into one-hot indicator columns.
#
# - Column selection: parsed from model/columns2Btransformed with EnforceInclusion/EnforceExclusion
#   semantics, restricted to string-like dtypes (string/object/category).
# - Naming: new columns are prefixed with the source column and '=' separator (e.g., "Region=West")
#   to avoid collisions when different columns share the same category label.
# - Missing values: not encoded (rows with NA get all zeros for that column’s dummies).
# - removeSources: if true, drops the original columns after expansion.
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory ID
FACTORY = "org.knime.base.node.preproc.columntrans2.One2ManyCol2NodeFactory"


# --------------------------------------------------------------------------------------------------
# settings.xml → Settings
# --------------------------------------------------------------------------------------------------

@dataclass
class OneHotSettings:
    include_names: List[str]
    exclude_names: List[str]
    enforce_option: str       # "EnforceInclusion" | "EnforceExclusion"
    remove_sources: bool


def _collect_name_list(root: ET._Element, key: str) -> List[str]:
    """
    Collect numeric-indexed entries from:
      model/columns2Btransformed/<key>  where key ∈ {"included_names", "excluded_names"}
    """
    base = first_el(
        root,
        ".//*[local-name()='config' and @key='model']"
        "/*[local-name()='config' and @key='columns2Btransformed']"
        f"/*[local-name()='config' and @key='{key}']"
    )
    if base is None:
        return []
    items: List[Tuple[int, str]] = []
    for k, v in iter_entries(base):
        if k.isdigit() and v is not None:
            try:
                items.append((int(k), v))
            except Exception:
                pass
    items.sort(key=lambda t: t[0])
    out: List[str] = []
    seen = set()
    for _, name in items:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def parse_onehot_settings(node_dir: Optional[Path]) -> OneHotSettings:
    if not node_dir:
        return OneHotSettings([], [], "EnforceInclusion", True)

    sp = node_dir / "settings.xml"
    if not sp.exists():
        return OneHotSettings([], [], "EnforceInclusion", True)

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")

    include_names = _collect_name_list(root, "included_names")
    exclude_names = _collect_name_list(root, "excluded_names")

    enforce_option = "EnforceInclusion"
    en = first(root,
        ".//*[local-name()='config' and @key='model']"
        "/*[local-name()='config' and @key='columns2Btransformed']"
        "/*[local-name()='entry' and @key='enforce_option']/@value"
    )
    if en:
        enforce_option = (en or "").strip()

    remove_sources = True
    if model is not None:
        rs = first(model, ".//*[local-name()='entry' and @key='removeSources']/@value")
        if rs is not None:
            remove_sources = str(rs).strip().lower() in {"true", "1", "yes", "y"}

    return OneHotSettings(
        include_names=include_names,
        exclude_names=exclude_names,
        enforce_option=enforce_option,
        remove_sources=remove_sources,
    )


# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    return ["import pandas as pd"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.columntrans2.One2ManyCol2NodeFactory"
)


def _emit_onehot_code(cfg: OneHotSettings) -> List[str]:
    """
    Returns python lines that transform df -> out_df by one-hot encoding.
    Selection logic:
      - Candidate columns = string-like dtypes (string/object/category)
      - If EnforceInclusion: intersect with include_names
      - If EnforceExclusion: candidate minus exclude_names
    Naming:
      - Uses prefix=<col>, prefix_sep='=' → '<col>=<label>'
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")

    # Determine candidate columns by dtype
    lines.append("str_like = out_df.select_dtypes(include=['string','object','category']).columns.tolist()")

    # Apply enforce semantics
    inc_list = "[" + ", ".join(repr(c) for c in (cfg.include_names or [])) + "]"
    exc_list = "[" + ", ".join(repr(c) for c in (cfg.exclude_names or [])) + "]"
    lines.append(f"_included_cfg = {inc_list}")
    lines.append(f"_excluded_cfg = {exc_list}")
    lines.append(f"_enforce = {repr(cfg.enforce_option or 'EnforceInclusion')}")

    lines.append("if _enforce == 'EnforceInclusion':")
    lines.append("    _targets = [c for c in _included_cfg if c in str_like and c in out_df.columns]")
    lines.append("else:  # EnforceExclusion")
    lines.append("    _targets = [c for c in str_like if c not in set(_excluded_cfg) and c in out_df.columns]")

    lines.append("if not _targets:")
    lines.append("    # nothing to transform")
    lines.append("    pass")
    lines.append("else:")
    # For each target column, build dummies with safe, unique names (prefix=col)
    lines.append("    dummy_frames = []")
    lines.append("    for _c in _targets:")
    lines.append("        _ser = out_df[_c].astype('string')  # keep NA as NA; dummy_na=False → all zeros row")
    lines.append("        _dm = pd.get_dummies(_ser, prefix=_c, prefix_sep='=', dummy_na=False)")
    lines.append("        dummy_frames.append(_dm)")
    lines.append("    if dummy_frames:")
    lines.append("        _dummies = pd.concat(dummy_frames, axis=1)")
    lines.append("        out_df = pd.concat([out_df, _dummies], axis=1)")
    # Remove sources if requested
    if cfg.remove_sources:
        lines.append("        out_df = out_df.drop(columns=_targets, errors='ignore')")
    else:
        lines.append("        # removeSources = false → keep originals")
        lines.append("        pass")

    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_onehot_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_onehot_code(cfg))

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
    return "\n".join(generate_py_body(node_id, node_dir, in_ports, out_ports)) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Returns (imports, body_lines) if this module can handle the node.
    """
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
