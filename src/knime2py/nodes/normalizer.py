#!/usr/bin/env python3

"""
Normalizer module for KNIME to Python conversion.

Overview
----------------------------
This module generates Python code to normalize selected columns using Min–Max or Z-Score
normalization methods based on settings defined in `settings.xml`. The generated code fits
into the knime2py generator pipeline, producing a DataFrame that is written to the node's
context.

Runtime Behavior
----------------------------
Inputs:
- The generated code reads a DataFrame from the context using the key format
  `context['<source_id>:<in_port>']`.

Outputs:
- The normalized DataFrame is written back to the context with the key format
  `context['<node_id>:<out_port>']`, where `<out_port>` defaults to '1'.

Key algorithms or mappings:
- The module implements Min-Max and Z-Score normalization techniques, handling numeric and
  boolean columns while excluding specified columns based on the configuration.

Edge Cases
----------------------------
The code includes safeguards for empty or constant columns, ensuring that they are handled
gracefully by mapping them to a default value or returning a zero vector for Z-Score
normalization. It also manages NaN values appropriately during the normalization process.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries: pandas. These dependencies
are necessary for the execution of the generated code, not for this module itself.

Usage
----------------------------
This module is typically invoked by the KNIME emitter for normalization nodes. An example
of expected context access is:
```python
df = context['source_id:1']  # input table
```

Node Identity
----------------------------
KNIME factory id: `FACTORY` is set to
`"org.knime.base.node.preproc.normalize3.Normalizer3NodeFactory"`.

Configuration
----------------------------
The module uses the `NormalizerSettings` dataclass for configuration, which includes the
following important fields:
- `mode`: Normalization method, defaults to "MINMAX".
- `new_min`: Minimum value for Min-Max normalization, defaults to 0.0.
- `new_max`: Maximum value for Min-Max normalization, defaults to 1.0.
- `excludes`: List of columns to exclude from normalization, populated from the settings.

The `parse_normalizer_settings` function extracts these values from the `settings.xml` file
using XPath queries, with fallbacks for missing configurations.

Limitations
----------------------------
The module does not support certain advanced normalization options available in KNIME and
may approximate behavior in some cases.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.node.preproc.normalize3.Normalizer3NodeFactory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory for Normalizer
FACTORY = "org.knime.base.node.preproc.normalize3.Normalizer3NodeFactory"

# --------------------------------------------------------------------------------------------------
# settings.xml → NormalizerSettings
# --------------------------------------------------------------------------------------------------

@dataclass
class NormalizerSettings:
    mode: str = "MINMAX"        # "MINMAX" or "ZSCORE"
    new_min: float = 0.0        # only for MINMAX
    new_max: float = 1.0        # only for MINMAX
    excludes: List[str] = field(default_factory=list)  # ONLY manuallyDeselected (or excluded_names fallback)

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
    
    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        NormalizerSettings: The parsed normalization settings.
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

    # -------- Normalization mode (exclude filter-config subtree) --------
    # Prefer explicit normalizationMethod if present (and outside filter-config)
    norm_mode = first(
        model,
        ".//*[local-name()='entry' and @key='normalizationMethod'"
        " and not(ancestor::*[local-name()='config' and (@key='dataColumnFilterConfig' or @key='data-column-filter')])"
        "]/@value"
    )

    if not norm_mode:
        norm_mode = first(
            model,
            ".//*[local-name()='entry' and @key='mode'"
            " and not(ancestor::*[local-name()='config' and (@key='dataColumnFilterConfig' or @key='data-column-filter')])"
            "]/@value"
        )

    mode = (norm_mode or "MINMAX").strip().upper()

    # -------- new-min / new-max (exclude filter-config subtree) --------
    def _float_entry_excl(keys: List[str], default: float) -> float:
        """Retrieve a float entry from the model, excluding certain subtrees."""
        for k in keys:
            v = first(
                model,
                f".//*[local-name()='entry' and @key='{k}']"
                " [not(ancestor::*[local-name()='config' and (@key='dataColumnFilterConfig' or @key='data-column-filter')])]/@value"
            )
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return default

    new_min = _float_entry_excl(["new-min", "newMin"], 0.0)
    new_max = _float_entry_excl(["new-max", "newMax"], 1.0)

    # -------- Column filter: only manuallyDeselected (fallback excluded_names) --------
    excludes: List[str] = []
    # Support both spellings
    dcf = first_el(model, ".//*[local-name()='config' and (@key='dataColumnFilterConfig' or @key='data-column-filter')]")
    if dcf is not None:
        man_desel = first_el(dcf, ".//*[local-name()='config' and @key='manualFilter']/*[local-name()='config' and @key='manuallyDeselected']")
        if man_desel is not None:
            excludes.extend(_collect_numeric_name_entries(man_desel))
        else:
            exc_old = first_el(dcf, ".//*[local-name()='config' and @key='excluded_names']")
            if exc_old is not None:
                excludes.extend(_collect_numeric_name_entries(exc_old))

    # Uniquify, preserve order
    excludes = list(dict.fromkeys([c for c in excludes if c]))

    return NormalizerSettings(
        mode=mode,
        new_min=new_min,
        new_max=new_max,
        excludes=excludes,
    )

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    """Generate the necessary imports for the normalization process."""
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.normalize3.Normalizer3NodeFactory"
)

# pandas dtype groups: keep broad to catch extension dtypes (nullable Int64/Float64) and bools
_NUMERIC_DTYPES = "['number', 'bool', 'boolean', 'Int64', 'Float64']"

def _emit_normalize_code(cfg: NormalizerSettings) -> List[str]:
    """
    Build `out_df` by:
      - starting from ALL columns, dropping only excludes
      - normalizing numeric/boolean columns among the remaining set
    
    Args:
        cfg (NormalizerSettings): The normalization settings to apply.

    Returns:
        List[str]: The lines of code to normalize the DataFrame.
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    lines.append("all_cols = out_df.columns.tolist()")
    if cfg.excludes:
        exc_list = ", ".join(repr(c) for c in cfg.excludes)
        lines.append(f"exclude_cols = [{exc_list}]")
        lines.append("cand_cols = [c for c in all_cols if c not in set(exclude_cols)]")
    else:
        lines.append("cand_cols = all_cols")

    # Numeric/boolean subset to normalize
    lines.append(f"norm_cols = out_df[cand_cols].select_dtypes(include={_NUMERIC_DTYPES}).columns.tolist()")

    lines.append("if not norm_cols:")
    lines.append("    # No numeric columns to normalize; passthrough")
    lines.append("    pass")
    lines.append("else:")
    lines.append("    # Coerce selected columns to numeric before normalization")
    lines.append("    out_df[norm_cols] = out_df[norm_cols].apply(pd.to_numeric, errors='coerce')")

    mode = (cfg.mode or "MINMAX").upper()

    if mode == "MINMAX":
        lines.append(f"    _new_min, _new_max = {cfg.new_min}, {cfg.new_max}")
        lines.append("    _span = (_new_max - _new_min)")
        lines.append("    _col_min = out_df[norm_cols].min(axis=0, skipna=True)")
        lines.append("    _col_max = out_df[norm_cols].max(axis=0, skipna=True)")
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
        lines.append("    def _zscore_col(s):")
        lines.append("        mu = _col_mean.get(s.name)")
        lines.append("        sd = _col_std.get(s.name)")
        lines.append("        if sd is None or pd.isna(sd) or sd == 0:")
        lines.append("            return pd.Series([0.0] * len(s), index=s.index)")
        lines.append("        return ((s - mu) / sd).astype(float)")
        lines.append("    out_df[norm_cols] = out_df[norm_cols].apply(_zscore_col)")
    else:
        # Should not happen now that we disambiguate modes, but keep a safe passthrough
        lines.append(f"    # Unsupported Normalizer mode '{cfg.mode}'; leaving columns unchanged")

    return lines

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python code body for the normalization process.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: The lines of code for the node's body.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_normalizer_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_normalize_code(cfg))

    # Publish (default port '1')
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
    """
    Generate the code for a Jupyter notebook cell.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        str: The generated code for the notebook cell.
    """
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"

def handle(ntype, nid, npath, incoming, outgoing):
    """
    Central entry used by emitters:
      - returns (imports, body_lines) if this module can handle the node type
      - returns None otherwise

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming connections.
        outgoing: The outgoing connections.

    Returns:
        tuple: A tuple containing the imports and body lines, or None if not handled.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
