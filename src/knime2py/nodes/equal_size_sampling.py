#!/usr/bin/env python3

"""
Equal Size Sampling.

Overview
----------------------------
This module generates Python code to perform equal size sampling on a DataFrame,
downsampling classes to the same size across a specified class/label column.

Runtime Behavior
----------------------------
Inputs:
- Reads a DataFrame from the context using the key format `'{src_id}:{in_port}'`.

Outputs:
- Writes the resulting DataFrame to the context using the key format `'{node_id}:{port}'`,
  where `port` is determined by the outgoing connections.

Key algorithms:
- Utilizes sklearn's `resample` function to achieve downsampling without replacement,
  ensuring reproducibility through a specified random seed.

Edge Cases
----------------------------
- Handles empty DataFrames by returning an empty DataFrame.
- Safeguards against constant columns and NaN values by checking group sizes before sampling.
- Implements logic to manage class imbalance by downsampling to the minimum class size.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries:
- pandas
- sklearn

These dependencies are required by the generated code, not by this module.

Usage
----------------------------
Typically invoked by the knime2py emitter for nodes that require equal size sampling.
Example context access:
```python
df = context['input_table:1']
```

Node Identity
----------------------------
KNIME factory id:
- FACTORY = "org.knime.base.node.preproc.equalsizesampling.EqualSizeSamplingNodeFactory"

Configuration
----------------------------
The settings are defined in the `EqualSizeSamplingSettings` dataclass, which includes:
- `class_col`: The name of the class/label column (default: None).
- `seed`: The random seed for reproducibility (default: 1).
- `method`: The sampling method, which is "Exact" (default: "Exact").

The `parse_equal_size_sampling_settings` function extracts these values from `settings.xml`
using XPath queries, with fallbacks to defaults as necessary.

Limitations
----------------------------
Approximate sampling is not implemented; only the "Exact" method is supported.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.node.preproc.equalsizesampling.EqualSizeSamplingNodeFactory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # normalize_in_ports, first/first_el, iter_entries, collect_module_imports, split_out_imports


# KNIME factory for Equal Size Sampling
FACTORY = "org.knime.base.node.preproc.equalsizesampling.EqualSizeSamplingNodeFactory"

# ---------------------------------------------------------------------
# settings.xml → EqualSizeSamplingSettings
# ---------------------------------------------------------------------

@dataclass
class EqualSizeSamplingSettings:
    class_col: Optional[str] = None
    seed: int = 1
    method: str = "Exact"   # KNIME exposes "Exact" vs "Approximate"; we implement Exact (downsample to min)


def parse_equal_size_sampling_settings(node_dir: Optional[Path]) -> EqualSizeSamplingSettings:
    """
    Parse Equal Size Sampling settings from the specified node directory.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        EqualSizeSamplingSettings: An instance containing the parsed settings.
    """
    if not node_dir:
        return EqualSizeSamplingSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return EqualSizeSamplingSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return EqualSizeSamplingSettings()

    class_col = first(model, ".//*[local-name()='entry' and @key='classColumn']/@value")
    method = (first(model, ".//*[local-name()='entry' and @key='samplingMethod']/@value") or "Exact").strip()
    seed_raw = first(model, ".//*[local-name()='entry' and @key='seed']/@value")
    try:
        seed = int(seed_raw) if seed_raw is not None else 1
    except Exception:
        seed = 1

    return EqualSizeSamplingSettings(class_col=class_col or None, seed=seed, method=method or "Exact")


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    """
    Generate the necessary import statements for the code.

    Returns:
        List[str]: A list of import statements.
    """
    # Use sklearn for the sampling
    return [
        "import pandas as pd",
        "from sklearn.utils import resample",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.equalsizesampling.EqualSizeSamplingNodeFactory"
)

def _emit_equal_size_code(cfg: EqualSizeSamplingSettings) -> List[str]:
    """
    Emit lines that create `out_df` with equal class sizes across `cfg.class_col`.

    Args:
        cfg (EqualSizeSamplingSettings): The settings for equal size sampling.

    Returns:
        List[str]: A list of code lines to perform equal size sampling.
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    if not cfg.class_col:
        lines.append("# No class column configured; passthrough.")
        return lines

    lines.append(f"_class = {repr(cfg.class_col)}")
    lines.append(f"_seed = {cfg.seed}")

    # Exact equal-size downsampling to the minimum class size using sklearn
    lines.append("if df.empty:")
    lines.append("    out_df = df.iloc[0:0]")
    lines.append("else:")
    lines.append("    groups = [g for _, g in df.groupby(_class, dropna=False, sort=False)]")
    lines.append("    if not groups:")
    lines.append("        out_df = df.iloc[0:0]")
    lines.append("    else:")
    lines.append("        min_count = min(len(g) for g in groups)")
    lines.append("        if min_count <= 0:")
    lines.append("            out_df = df.iloc[0:0]")
    lines.append("        else:")
    lines.append("            parts = [resample(g, replace=False, n_samples=min_count, random_state=_seed) for g in groups]")
    lines.append("            out_df = pd.concat(parts, axis=0).sort_index() if parts else df.iloc[0:0]")

    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python code body for the node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: A list of code lines representing the body of the node.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_equal_size_sampling_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_equal_size_code(cfg))

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
    """
    Generate the code for a Jupyter notebook cell.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        str: The generated code as a string.
    """
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Entry used by emitters to handle the node type.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming connections.
        outgoing: The outgoing connections.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines if the node type is handled, None otherwise.
    """

    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
