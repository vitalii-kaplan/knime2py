#!/usr/bin/env python3

"""
One-Hot Encoding for KNIME-style DataFrames.

Overview
----------------------------
This module generates Python code for one-hot encoding of categorical variables
in a DataFrame, following the naming and placement conventions of KNIME.

Runtime Behavior
----------------------------
Inputs:
- Reads a DataFrame from the context using the key format `context['<source_id>:<port>']`.

Outputs:
- Writes the transformed DataFrame back to the context with the key format
  `context['<node_id>:<port>']`, where `<port>` is the output port number.

Key algorithms:
- The module implements one-hot encoding with KNIME-style naming, where each
  new column is named as `<level>_<Column>`, preserving the order of first
  appearance in the data.

Edge Cases
----------------------------
The code handles empty or constant columns by skipping transformation. It also
produces all-zero indicators for columns with missing values.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries:
- pandas

These dependencies are required by the generated code, not by this module.

Usage
----------------------------
This module is typically invoked by the knime2py emitter as part of the code
generation process for KNIME nodes. An example of context access is:
```python
df = context['source_id:1']  # input table
```

Node Identity
----------------------------
KNIME factory id:
- FACTORY = "org.knime.base.node.preproc.columntrans2.One2ManyCol2NodeFactory"

Configuration
----------------------------
The settings are defined in the `OneHotSettings` dataclass, which includes:
- `include_names`: List of columns to include in the transformation.
- `exclude_names`: List of columns to exclude from the transformation.
- `enforce_option`: Determines whether to enforce inclusion or exclusion of columns.
- `remove_sources`: Indicates whether to drop the original source columns.

The `parse_onehot_settings` function extracts these values from the `settings.xml`
file, with appropriate fallbacks.

Limitations
----------------------------
This implementation does not support all KNIME features, such as advanced
collision handling or specific configurations not covered by the settings.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.node.preproc.columntrans2.One2ManyCol2NodeFactory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

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
    
    Args:
        root (ET._Element): The root XML element to search within.
        key (str): The key to look for in the XML structure.

    Returns:
        List[str]: A list of unique names collected from the specified key.
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
    """
    Parse the one-hot encoding settings from the settings.xml file.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        OneHotSettings: An instance of OneHotSettings populated with the parsed values.
    """
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
    en = first(
        root,
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
    """
    Generate the necessary imports for the one-hot encoding code.

    Returns:
        List[str]: A list of import statements.
    """
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.columntrans2.One2ManyCol2NodeFactory"
)

def _emit_onehot_code(cfg: OneHotSettings) -> List[str]:
    """
    Transform df -> out_df with KNIME-style naming and placement.

    Args:
        cfg (OneHotSettings): The configuration settings for one-hot encoding.

    Returns:
        List[str]: A list of code lines that perform the one-hot encoding transformation.
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")

    # 1) Determine string-like candidates (preserves table order)
    lines.append("str_like = [c for c in out_df.columns if out_df[c].dtype.name in ('string','object','category')]")

    # 2) Target column order per enforce semantics
    inc_list = "[" + ", ".join(repr(c) for c in (cfg.include_names or [])) + "]"
    exc_list = "[" + ", ".join(repr(c) for c in (cfg.exclude_names or [])) + "]"
    lines.append(f"_included_cfg = {inc_list}")
    lines.append(f"_excluded_cfg = {exc_list}")
    lines.append(f"_enforce = {repr(cfg.enforce_option or 'EnforceInclusion')}")

    lines.append("if _enforce == 'EnforceInclusion':")
    lines.append("    _targets = [c for c in _included_cfg if c in str_like and c in out_df.columns]")
    lines.append("else:  # EnforceExclusion → keep table order")
    lines.append("    _targets = [c for c in out_df.columns if (c in str_like) and (c not in set(_excluded_cfg))]")

    lines.append("if not _targets:")
    lines.append("    # nothing to transform")
    lines.append("    pass")
    lines.append("else:")
    # 3) Build dummies for each target with KNIME-style names and first-appearance level order
    lines.append("    _dummy_frames = []   # preserve target ordering for block order")
    lines.append("    for _c in _targets:")
    lines.append("        s = out_df[_c].astype('string')")
    lines.append("        # level order by first appearance (exclude NA)")
    lines.append("        levels = s.dropna().drop_duplicates().tolist()")
    lines.append("        cols = {}")
    lines.append("        for lv in levels:")
    lines.append("            cname = f\"{lv}_{_c}\"  # VALUE_first")
    lines.append("            cols[cname] = (s == lv).astype('float64')")
    lines.append("        _dm = pd.DataFrame(cols, index=s.index) if cols else pd.DataFrame(index=s.index)")
    lines.append("        _dummy_frames.append(_dm)")

    # 4) Base table: either drop sources (removeSources) or keep them; always append dummies at END
    if cfg.remove_sources:
        lines.append("    base_cols = [c for c in out_df.columns if c not in set(_targets)]")
    else:
        lines.append("    base_cols = list(out_df.columns)")
    lines.append("    base_df = out_df[base_cols].copy()")
    lines.append("    dummies_df = pd.concat(_dummy_frames, axis=1) if _dummy_frames else pd.DataFrame(index=out_df.index)")
    lines.append("    out_df = pd.concat([base_df, dummies_df], axis=1)")

    return lines

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python code body for the one-hot encoding node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports for the node.
        out_ports (Optional[List[str]]): The outgoing ports for the node.

    Returns:
        List[str]: A list of code lines that represent the body of the node.
    """
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
    """
    Generate the code for a Jupyter notebook cell.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports for the node.
        out_ports (Optional[List[str]]): The outgoing ports for the node.

    Returns:
        str: The generated code as a string.
    """
    return "\n".join(generate_py_body(node_id, node_dir, in_ports, out_ports)) + "\n"

def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handle the node processing and return the necessary imports and body lines.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path to the node.
        incoming: The incoming connections to the node.
        outgoing: The outgoing connections from the node.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the list of imports and the body lines.
    """
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
