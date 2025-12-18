#!/usr/bin/env python3

"""
Shared helpers for Normalizer-based nodes.

Contains:
    - NormalizerSettings dataclass
    - settings.xml parsers
    - code-emission utilities that normalize numeric columns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import first, first_el, iter_entries

__all__ = [
    "NormalizerSettings",
    "parse_normalizer_settings",
    "emit_normalize_code",
]


@dataclass
class NormalizerSettings:
    mode: str = "MINMAX"        # "MINMAX" or "ZSCORE"
    new_min: float = 0.0        # only for MINMAX
    new_max: float = 1.0        # only for MINMAX
    excludes: List[str] = field(default_factory=list)  # excluded columns
    columns: List[str] = field(default_factory=list)   # explicit columns to normalize
    all_numeric: bool = False                         # normalize all numeric columns


def _collect_numeric_name_entries(cfg: ET._Element) -> List[str]:
    """Collect <entry key='0' value='col'>, <entry key='1' value='col'> … under cfg."""
    out: List[str] = []
    for k, v in iter_entries(cfg):
        if k.isdigit() and v:
            out.append(v.strip())
    return out


def parse_normalizer_settings(node_dir: Optional[Path]) -> NormalizerSettings:
    """
    Parse Normalizer settings with disambiguation:
      - Normalization mode: model/entry[@key='mode'] (or 'normalizationMethod'), but NOT inside
        dataColumnFilterConfig / data-column-filter.
      - new-min/new-max (or newMin/newMax) also taken outside the filter-config subtree.
      - Column filter: ONLY read manualFilter/manuallyDeselected (fallback: excluded_names).
    """
    if not node_dir:
        return NormalizerSettings()
    sp = node_dir / "settings.xml"
    if not sp.exists():
        return NormalizerSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return NormalizerSettings()

    norm_mode = first(
        model,
        ".//*[local-name()='entry' and @key='normalizationMethod'"
        " and not(ancestor::*[local-name()='config' and (@key='dataColumnFilterConfig' or @key='data-column-filter')])"
        "]/@value",
    )

    if not norm_mode:
        norm_mode = first(
            model,
            ".//*[local-name()='entry' and @key='mode'"
            " and not(ancestor::*[local-name()='config' and (@key='dataColumnFilterConfig' or @key='data-column-filter')])"
            "]/@value",
        )

    raw_mode = (norm_mode or "MINMAX").strip()
    mode = raw_mode.upper()
    if mode not in {"MINMAX", "ZSCORE"}:
        if raw_mode in {"0", "1"}:
            mode = "ZSCORE"
        else:
            mode = "MINMAX"

    def _float_entry_excl(keys: List[str], default: float) -> float:
        """Retrieve a float entry from the model, excluding certain subtrees."""
        for k in keys:
            v = first(
                model,
                f".//*[local-name()='entry' and @key='{k}']"
                " [not(ancestor::*[local-name()='config' and (@key='dataColumnFilterConfig' or @key='data-column-filter')])]/@value",
            )
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return default

    new_min = _float_entry_excl(["new-min", "newMin"], 0.0)
    new_max = _float_entry_excl(["new-max", "newMax"], 1.0)

    excludes: List[str] = []
    dcf = first_el(
        model,
        ".//*[local-name()='config' and (@key='dataColumnFilterConfig' or @key='data-column-filter')]",
    )
    if dcf is not None:
        man_desel = first_el(
            dcf,
            ".//*[local-name()='config' and @key='manualFilter']"
            "/*[local-name()='config' and @key='manuallyDeselected']",
        )
        if man_desel is not None:
            excludes.extend(_collect_numeric_name_entries(man_desel))
        else:
            exc_old = first_el(dcf, ".//*[local-name()='config' and @key='excluded_names']")
            if exc_old is not None:
                excludes.extend(_collect_numeric_name_entries(exc_old))

    excludes = list(dict.fromkeys([c for c in excludes if c]))

    # Optional explicit columns list
    columns: List[str] = []
    cols_cfg = first_el(
        model,
        ".//*[local-name()='config' and @key='columns']",
    )
    if cols_cfg is not None:
        columns = _collect_numeric_name_entries(cols_cfg)

    # Flag for "all numeric columns"
    all_numeric_entry = first(
        model,
        ".//*[local-name()='entry' and @key='all_numeric_columns_used']/@value",
    )
    all_numeric = str(all_numeric_entry or "").strip().lower() == "true"

    return NormalizerSettings(
        mode=mode,
        new_min=new_min,
        new_max=new_max,
        excludes=excludes,
        columns=columns,
        all_numeric=all_numeric,
    )


_NUMERIC_DTYPES = "['number', 'bool', 'boolean', 'Int64', 'Float64']"


def emit_normalize_code(cfg: NormalizerSettings, bundle_var: str = "bundle") -> List[str]:
    """
    Emit Python code lines that normalize numeric columns and capture stats.

    Returns:
        List[str]: Lines of code to append to the generated body.
    """
    mode = (cfg.mode or "MINMAX").upper()
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    lines.append(f"{bundle_var} = {{")
    lines.append(f"    'mode': {repr(mode)},")
    lines.append(f"    'new_min': {cfg.new_min},")
    lines.append(f"    'new_max': {cfg.new_max},")
    lines.append(f"    'excludes': {repr(cfg.excludes)},")
    lines.append("    'columns': [],")
    lines.append("    'stats': {},")
    lines.append("}")
    lines.append("all_cols = out_df.columns.tolist()")
    if cfg.columns:
        lines.append(f"cand_cols = [c for c in {repr(cfg.columns)} if c in all_cols]")
    elif cfg.all_numeric:
        lines.append("cand_cols = out_df.select_dtypes(include=['number']).columns.tolist()")
    else:
        lines.append("cand_cols = all_cols")
    if cfg.excludes:
        exc_list = ", ".join(repr(c) for c in cfg.excludes)
        lines.append(f"exclude_cols = [{exc_list}]")
        lines.append("cand_cols = [c for c in cand_cols if c not in set(exclude_cols)]")

    lines.append(f"norm_cols = out_df[cand_cols].select_dtypes(include={_NUMERIC_DTYPES}).columns.tolist()")
    lines.append(f"{bundle_var}['columns'] = list(norm_cols)")

    lines.append("if not norm_cols:")
    lines.append("    # No numeric columns to normalize; passthrough")
    lines.append(f"    {bundle_var}['stats'] = {{}}")
    lines.append("    pass")
    lines.append("else:")
    lines.append("    # Coerce selected columns to numeric before normalization")
    lines.append("    out_df[norm_cols] = out_df[norm_cols].apply(pd.to_numeric, errors='coerce')")
    if mode == "MINMAX":
        lines.append(f"    _new_min, _new_max = {cfg.new_min}, {cfg.new_max}")
        lines.append("    _span = (_new_max - _new_min)")
        lines.append("    _col_min = out_df[norm_cols].min(axis=0, skipna=True)")
        lines.append("    _col_max = out_df[norm_cols].max(axis=0, skipna=True)")
        lines.append("    stats = {}")
        lines.append("    for col in norm_cols:")
        lines.append("        mn = _col_min.get(col)")
        lines.append("        mx = _col_max.get(col)")
        lines.append("        stats[col] = {")
        lines.append("            'min': None if pd.isna(mn) else float(mn),")
        lines.append("            'max': None if pd.isna(mx) else float(mx),")
        lines.append("        }")
        lines.append(f"    {bundle_var}['stats'] = stats")
        lines.append("    def _minmax_col(s):")
        lines.append("        mn = _col_min.get(s.name)")
        lines.append("        mx = _col_max.get(s.name)")
        lines.append("        rng = (mx - mn) if (mn is not None and mx is not None) else None")
        lines.append("        if rng is None or pd.isna(rng) or rng == 0:")
        lines.append("            # constant/empty column → map to new_min")
        lines.append("            return pd.Series([_new_min] * len(s), index=s.index)")
        lines.append("        return (_new_min + (s - mn) / rng * _span).astype(float)")
        lines.append("    out_df[norm_cols] = out_df[norm_cols].apply(_minmax_col)")
    elif mode == "ZSCORE":
        lines.append("    _col_mean = out_df[norm_cols].mean(axis=0, skipna=True)")
        lines.append("    _col_std  = out_df[norm_cols].std(axis=0, ddof=0, skipna=True)")
        lines.append("    stats = {}")
        lines.append("    for col in norm_cols:")
        lines.append("        mu = _col_mean.get(col)")
        lines.append("        sd = _col_std.get(col)")
        lines.append("        stats[col] = {")
        lines.append("            'mean': None if pd.isna(mu) else float(mu),")
        lines.append("            'std': None if pd.isna(sd) else float(sd),")
        lines.append("        }")
        lines.append(f"    {bundle_var}['stats'] = stats")
        lines.append("    def _zscore_col(s):")
        lines.append("        mu = _col_mean.get(s.name)")
        lines.append("        sd = _col_std.get(s.name)")
        lines.append("        if sd is None or pd.isna(sd) or sd == 0:")
        lines.append("            return pd.Series([0.0] * len(s), index=s.index)")
        lines.append("        return ((s - mu) / sd).astype(float)")
        lines.append("    out_df[norm_cols] = out_df[norm_cols].apply(_zscore_col)")
    else:
        lines.append(f"    # Unsupported Normalizer mode '{cfg.mode}'; leaving columns unchanged")

    return lines
