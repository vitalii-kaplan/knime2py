#!/usr/bin/env python3

####################################################################################################
#
# Value Lookup
#
# Joins a "data" table (left) with a "dictionary" table (right) to look up values by key.
# Parsed from settings.xml:
#   - lookupCol (left key), dictKeyCol (right key)
#   - dictValueCols (which right-side value columns to bring in)
#   - caseSensitive (bool)
#   - lookupReplacementCol (optional: replace/overlay a left column with looked-up values)
#   - columnNoMatchReplacement ∈ {'RETAIN', ...}  (RETAIN keeps original where no match)
#   - lookupColumnOutput ∈ {'RETAIN','REMOVE'} (remove left key if requested)
#   - createFoundCol (bool) — add boolean "Found" column (any looked-up col non-null)
#
# Merge details:
#   - To avoid dtype mismatches we always cast join keys to pandas 'string' dtype.
#   - If caseSensitive is False we compare lowercased string keys.
#   - We avoid name collisions by suffixing new columns with "_lkp" when needed.
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (
    first,
    first_el,
    iter_entries,
    normalize_in_ports,
    collect_module_imports,
    split_out_imports,
)

FACTORY = "org.knime.base.node.preproc.valuelookup.ValueLookupNodeFactory"

# ---------------------------------------------------------------------
# settings.xml → ValueLookupSettings
# ---------------------------------------------------------------------

@dataclass
class ValueLookupSettings:
    lookup_col: Optional[str] = None
    dict_key_col: Optional[str] = None
    dict_value_cols: List[str] = field(default_factory=list)
    case_sensitive: bool = True
    replace_col: Optional[str] = None
    no_match: str = "RETAIN"          # behavior on no-match; we honor RETAIN specifically
    lookup_col_output: str = "RETAIN" # 'RETAIN' | 'REMOVE'
    create_found_col: bool = False


def _bool(s: Optional[str], default: bool) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def _collect_numeric_entry_values(root: ET._Element) -> List[str]:
    """
    Collect values from <entry key="0" value="..."/>, <entry key="1" value="..."/>, ...
    anywhere under the given root node.
    """
    out: List[str] = []
    for k, v in iter_entries(root):
        if not k:
            continue
        if k.isdigit() and v:
            out.append(v)
    return out


def parse_lookup_settings(node_dir: Optional[Path]) -> ValueLookupSettings:
    if not node_dir:
        return ValueLookupSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return ValueLookupSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return ValueLookupSettings()

    lookup_col = first(model_el, ".//*[local-name()='entry' and @key='lookupCol']/@value")
    dict_key_col = first(model_el, ".//*[local-name()='entry' and @key='dictKeyCol']/@value")

    # Value columns: prefer explicit list if present
    dict_vals_el = first_el(model_el, ".//*[local-name()='config' and @key='dictValueCols']")
    value_cols: List[str] = []
    if dict_vals_el is not None:
        # try 'selected_Internals', 'manualFilter/manuallySelected', or any digit-keys under the block
        sel_int = first_el(dict_vals_el, ".//*[local-name()='config' and @key='selected_Internals']")
        if sel_int is not None:
            value_cols.extend(_collect_numeric_entry_values(sel_int))
        man_sel = first_el(
            dict_vals_el,
            ".//*[local-name()='config' and @key='manualFilter']/*[local-name()='config' and @key='manuallySelected']",
        )
        if man_sel is not None:
            value_cols.extend(_collect_numeric_entry_values(man_sel))
        if not value_cols:
            value_cols.extend(_collect_numeric_entry_values(dict_vals_el))

    # Flags and other options
    case_sensitive = _bool(first(model_el, ".//*[local-name()='entry' and @key='caseSensitive']/@value"), True)
    replace_col = first(model_el, ".//*[local-name()='entry' and @key='lookupReplacementCol']/@value")
    no_match = first(model_el, ".//*[local-name()='entry' and @key='columnNoMatchReplacement']/@value") or "RETAIN"
    lookup_out = first(model_el, ".//*[local-name()='entry' and @key='lookupColumnOutput']/@value") or "RETAIN"
    create_found = _bool(first(model_el, ".//*[local-name()='entry' and @key='createFoundCol']/@value"), False)

    # De-duplicate while preserving order; never include the dictionary key itself
    seen = set()
    dedup_vals: List[str] = []
    for c in value_cols:
        if not c or c == dict_key_col:
            continue
        if c not in seen:
            seen.add(c)
            dedup_vals.append(c)

    return ValueLookupSettings(
        lookup_col=lookup_col or None,
        dict_key_col=dict_key_col or None,
        dict_value_cols=dedup_vals,
        case_sensitive=case_sensitive,
        replace_col=(replace_col or None),
        no_match=no_match,
        lookup_col_output=lookup_out,
        create_found_col=create_found,
    )

# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    return ["import pandas as pd"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.valuelookup.ValueLookupNodeFactory"
)


def _emit_lookup_code(cfg: ValueLookupSettings) -> List[str]:
    """
    Emit python lines that join df_left with df_right, add columns, and optionally replace.
    """
    lines: List[str] = []
    lines.append("out_df = df_left.copy()")
    lines.append("dict_df = df_right.copy()")
    lines.append("if not (cfg_lookup and cfg_dict_key):")
    lines.append("    pass  # passthrough if keys not configured")
    lines.append("else:")
    # Decide which dictionary columns to bring in; never include the join key itself
    lines.append("    if cfg_value_cols:")
    lines.append("        value_cols = [c for c in cfg_value_cols if c and c != cfg_dict_key]")
    lines.append("    else:")
    lines.append("        value_cols = [c for c in dict_df.columns if c != cfg_dict_key]")
    # Build rename plan to avoid collisions; special handling for replacement column
    lines.append("    rename_map = {}")
    lines.append("    added_cols = []")
    lines.append("    replace_col_tmp = None")
    lines.append("    if cfg_replace_col and cfg_replace_col in value_cols:")
    lines.append("        replace_col_tmp = '__lkp_replace__'")
    lines.append("        rename_map[cfg_replace_col] = replace_col_tmp")
    lines.append("    for c in value_cols:")
    lines.append("        if c == cfg_replace_col:")
    lines.append("            continue")
    lines.append("        new_name = c if c not in out_df.columns else f'{c}_lkp'")
    lines.append("        rename_map[c] = new_name")
    lines.append("        added_cols.append(new_name)")
    # Normalize keys to string (avoid dtype mismatch); lowercase if not case-sensitive
    lines.append("    _k_left = '__lk_left__'; _k_right='__lk_right__'")
    lines.append("    if cfg_case_sensitive:")
    lines.append("        out_df[_k_left]  = out_df[cfg_lookup].astype('string')")
    lines.append("        dict_df[_k_right] = dict_df[cfg_dict_key].astype('string')")
    lines.append("    else:")
    lines.append("        out_df[_k_left]  = out_df[cfg_lookup].astype('string').str.lower()")
    lines.append("        dict_df[_k_right] = dict_df[cfg_dict_key].astype('string').str.lower()")
    # Right subset with normalized key and value columns; drop the original right key before merge
    lines.append("    right_sub = dict_df[[cfg_dict_key, _k_right] + value_cols].copy()")
    lines.append("    if rename_map:")
    lines.append("        right_sub = right_sub.rename(columns=rename_map)")
    lines.append("    right_sub = right_sub.drop(columns=[cfg_dict_key], errors='ignore')")
    # Merge on temp normalized keys; clean up temp columns
    lines.append("    merged = out_df.merge(right_sub, how='left', left_on=_k_left, right_on=_k_right)")
    lines.append("    merged = merged.drop(columns=[_k_left, _k_right], errors='ignore')")
    # Replacement behavior
    lines.append("    if replace_col_tmp is not None:")
    lines.append("        if cfg_replace_col not in merged.columns:")
    lines.append("            merged[cfg_replace_col] = merged[replace_col_tmp]")
    lines.append("        else:")
    lines.append("            if (cfg_no_match or 'RETAIN').upper() == 'RETAIN':")
    lines.append("                merged[cfg_replace_col] = merged[replace_col_tmp].where(")
    lines.append("                    merged[replace_col_tmp].notna(), merged[cfg_replace_col])")
    lines.append("            else:")
    lines.append("                merged[cfg_replace_col] = merged[replace_col_tmp]")
    lines.append("        merged = merged.drop(columns=[replace_col_tmp], errors='ignore')")
    # Found column (optional)
    lines.append("    if cfg_create_found:")
    lines.append("        _found_cols = list(added_cols)")
    lines.append("        if cfg_replace_col in merged.columns:")
    lines.append("            _found_cols.append(cfg_replace_col)")
    lines.append("        merged['Found'] = merged[_found_cols].notna().any(axis=1) if _found_cols else False")
    # Optionally remove the lookup column
    lines.append("    if (cfg_lookup_out or '').upper() == 'REMOVE':")
    lines.append("        merged = merged.drop(columns=[cfg_lookup], errors='ignore')")
    lines.append("    out_df = merged")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],   # two inputs: Port 1=data, Port 2=dictionary
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_lookup_settings(ndir)

    # Resolve inputs. Prefer mapping by *target* port if provided, else keep order.
    # Expect: target port '1' → data, '2' → dictionary.
    pairs: List[Tuple[str, str]] = []
    # normalize incoming to (src_id, src_port, tgt_port)
    for src_id, e in (in_ports or []):
        # in `handle` we pass tuples; this function expects normalized (src_id, src_port)
        # so we only use what we receive
        pass

    # In this generator, we only receive (src_id, src_port). Determine left/right by position.
    norm = normalize_in_ports(in_ports)
    left_src, left_in = norm[0] if norm else ("UNKNOWN", "1")
    right_src, right_in = (norm[1] if len(norm) > 1 else ("UNKNOWN", "1"))

    ports = out_ports or ["1"]

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"df_left  = context['{left_src}:{left_in}']   # data table")
    lines.append(f"df_right = context['{right_src}:{right_in}']  # dictionary table")
    lines.append(f"cfg_lookup = {repr(cfg.lookup_col)}")
    lines.append(f"cfg_dict_key = {repr(cfg.dict_key_col)}")
    lines.append(f"cfg_value_cols = {repr(list(cfg.dict_value_cols))}")
    lines.append(f"cfg_case_sensitive = {('True' if cfg.case_sensitive else 'False')}")
    lines.append(f"cfg_replace_col = {repr(cfg.replace_col)}")
    lines.append(f"cfg_no_match = {repr(cfg.no_match)}")
    lines.append(f"cfg_lookup_out = {repr(cfg.lookup_col_output)}")
    lines.append(f"cfg_create_found = {('True' if cfg.create_found_col else 'False')}")

    lines.extend(_emit_lookup_code(cfg))

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
    Returns (imports, body_lines) if this module can handle the node.

    Port mapping:
      - target port 1 → data
      - target port 2 → dictionary
    """

    explicit_imports = collect_module_imports(generate_imports)

    # Prefer target-port mapping when available
    left_pair: Optional[Tuple[str, str]] = None   # data
    right_pair: Optional[Tuple[str, str]] = None  # dictionary
    for src_id, e in (incoming or []):
        src_port = str(getattr(e, "source_port", "") or "1")
        tgt_port = str(getattr(e, "target_port", "") or "")
        if tgt_port == "1":
            left_pair = (str(src_id), src_port)
        elif tgt_port == "2":
            right_pair = (str(src_id), src_port)

    norm_in: List[Tuple[str, str]] = []
    if left_pair:
        norm_in.append(left_pair)
    if right_pair:
        norm_in.append(right_pair)
    if not norm_in:
        # fallback to original order if target ports were absent
        norm_in = [(str(src), str(getattr(e, "source_port", "") or "1")) for src, e in (incoming or [])]

    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, norm_in, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
