#!/usr/bin/env python3

"""
SMOTE implementation for oversampling minority classes.

Overview
----------------------------
This module emits Python code to perform SMOTE (Synthetic Minority Over-sampling Technique)
on input DataFrames, producing a resampled table that is written to the context. It fits
into the knime2py generator pipeline by translating KNIME node configurations into Python
code.

Runtime Behavior
----------------------------
Inputs:
- Reads a DataFrame from the context using the key format 'src_id:in_port'.

Outputs:
- Writes the resulting DataFrame to the context with the key format 'node_id:out_port',
  where out_port defaults to '1'.

Key algorithms:
- Utilizes imbalanced-learn's SMOTE for generating synthetic samples.
- Selects numeric and boolean columns as features, excluding the target column.
- Implements a sampling strategy based on the configured method and rate.

Edge Cases
----------------------------
The code handles various edge cases, including:
- Absence of a target column, resulting in a passthrough.
- Single-class scenarios or insufficient samples in the minority class, leading to fallback
  to the original DataFrame.
- NaN values and constant columns are managed by the underlying library.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries:
- pandas
- numpy
- imblearn
These dependencies are required for the generated code, not for this module itself.

Usage
----------------------------
This module is typically invoked by the knime2py emitter for nodes that require SMOTE
functionality. An example of expected context access is:
```python
df = context['{src_id}:{in_port}']  # input table
```

Node Identity
----------------------------
KNIME factory id:
- FACTORY = "org.knime.base.node.mine.smote.SmoteNodeFactory"

Configuration
----------------------------
The module uses the `SmoteSettings` dataclass for configuration, which includes:
- target: The class/target column (default: None).
- k_neighbors: The number of neighbors for kNN (default: 5).
- method: The sampling method (default: "oversample_equal").
- rate: The target ratio for sampling (default: 2.0).
- random_state: The seed for random number generation (default: 1).

The `parse_smote_settings` function extracts these values from the settings.xml file using
XPath queries, with sensible fallbacks for missing values.

Limitations
----------------------------
Currently, the implementation does not support all configurations available in KNIME,
and approximations may occur in behavior compared to the original node.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.node.mine.smote.SmoteNodeFactory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory
FACTORY = "org.knime.base.node.mine.smote.SmoteNodeFactory"


def can_handle(node_type: Optional[str]) -> bool:
    """Check if the given node type can be handled by this module.

    Args:
        node_type (Optional[str]): The type of the node.

    Returns:
        bool: True if the node type can be handled, False otherwise.
    """
    return bool(node_type and node_type.endswith(SMOTE_FACTORY))


# ---------------------------------------------------------------------
# settings.xml → SmoteSettings
# ---------------------------------------------------------------------

@dataclass
class SmoteSettings:
    target: Optional[str] = None      # class / target column
    k_neighbors: int = 5              # KNIME kNN
    method: str = "oversample_equal"  # KNIME 'method'
    rate: float = 2.0                 # KNIME 'rate' (used when method != oversample_equal)
    random_state: int = 1             # KNIME 'seed' (fallback to 1)


def _to_int(s: Optional[str], default: int) -> int:
    """Convert a string to an integer, returning a default value if conversion fails.

    Args:
        s (Optional[str]): The string to convert.
        default (int): The default value to return on failure.

    Returns:
        int: The converted integer or the default value.
    """
    try:
        return int(s) if s is not None else default
    except Exception:
        return default


def _to_float(s: Optional[str], default: float) -> float:
    """Convert a string to a float, returning a default value if conversion fails.

    Args:
        s (Optional[str]): The string to convert.
        default (float): The default value to return on failure.

    Returns:
        float: The converted float or the default value.
    """
    try:
        return float(s) if s is not None else default
    except Exception:
        return default


def parse_smote_settings(node_dir: Optional[Path]) -> SmoteSettings:
    """Parse the SMOTE settings from the settings.xml file.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        SmoteSettings: An instance of SmoteSettings populated with values from the XML.
    """
    if not node_dir:
        return SmoteSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return SmoteSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return SmoteSettings()

    k_neighbors = _to_int(first(model_el, ".//*[local-name()='entry' and @key='kNN']/@value"), 5)
    method = (first(model_el, ".//*[local-name()='entry' and @key='method']/@value") or "oversample_equal").strip()
    rate = _to_float(first(model_el, ".//*[local-name()='entry' and @key='rate']/@value"), 2.0)
    target = first(model_el, ".//*[local-name()='entry' and @key='class']/@value")

    seed_raw = first(model_el, ".//*[local-name()='entry' and @key='seed']/@value")
    try:
        seed = int(seed_raw) if seed_raw not in (None, "",) else 1
    except Exception:
        seed = 1

    return SmoteSettings(
        target=target or None,
        k_neighbors=max(1, k_neighbors),
        method=method or "oversample_equal",
        rate=rate,
        random_state=seed,
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    """Generate the necessary import statements for the SMOTE code.

    Returns:
        List[str]: A list of import statements.
    """
    return [
        "import pandas as pd",
        "import numpy as np",
        "from imblearn.over_sampling import SMOTE",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.smote.SmoteNodeFactory"
)


def _emit_smote_code(cfg: SmoteSettings) -> List[str]:
    """Emit the code lines necessary to perform SMOTE on the input DataFrame.

    Args:
        cfg (SmoteSettings): The configuration settings for SMOTE.

    Returns:
        List[str]: A list of code lines to perform SMOTE.
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    if not cfg.target:
        lines.append("# No target column configured; passthrough.")
        return lines

    # Choose numeric/bool features for SMOTE (SMOTE assumes continuous features).
    lines.append(f"_target = {repr(cfg.target)}")
    lines.append("num_like = df.select_dtypes(include=['number','bool','boolean','Int64','Float64']).columns.tolist()")
    lines.append("feat_cols = [c for c in num_like if c != _target]")
    lines.append("if len(feat_cols) == 0:")
    lines.append("    # Nothing to synthesize from; passthrough")
    lines.append("    out_df = df.copy()")
    lines.append("else:")
    lines.append("    X = df[feat_cols].copy()")
    lines.append("    y = df[_target].copy()")

    # Build sampling_strategy
    if cfg.method.strip().lower() == "oversample_equal":
        lines.append("    sampling_strategy = 'auto'  # equalize minorities to majority")
    else:
        lines.append(f"    _rate = float({float(cfg.rate)})")
        lines.append("    vc = y.value_counts(dropna=False)")
        lines.append("    if len(vc) <= 1:")
        lines.append("        # Single class or empty; passthrough")
        lines.append("        out_df = df.copy()")
        lines.append("    else:")
        lines.append("        maj_label = vc.idxmax()")
        lines.append("        maj_n = int(vc.max())")
        lines.append("        sampling_strategy = {}")
        lines.append("        for cls, cnt in vc.items():")
        lines.append("            if cls == maj_label:")
        lines.append("                continue")
        lines.append("            if 0.0 < _rate <= 1.0:")
        lines.append("                target_n = int(round(_rate * maj_n))")
        lines.append("            else:")
        lines.append("                target_n = int(round(_rate * int(cnt)))")
        lines.append("            if target_n > int(cnt):")
        lines.append("                sampling_strategy[cls] = target_n")
        lines.append("        if not sampling_strategy:")
        lines.append("            # Nothing to upsample → fallback to passthrough")
        lines.append("            out_df = df.copy()")

    # Effective k (avoid k > minority_count - 1 which raises in imblearn)
    lines.append(f"    _k = int({int(cfg.k_neighbors)})")
    lines.append("    min_class_count = int(y.value_counts(dropna=False).min()) if len(y) else 0")
    lines.append("    if min_class_count <= 1:")
    lines.append("        # SMOTE cannot operate with a single sample in minority; passthrough")
    lines.append("        out_df = df.copy()")
    lines.append("    else:")
    lines.append("        _k_eff = max(1, min(_k, min_class_count - 1))")

    # Random state
    lines.append(f"    _seed = int({int(cfg.random_state)})")

    # Run SMOTE
    lines.append("    try:")
    if cfg.method.strip().lower() == "oversample_equal":
        lines.append("        sm = SMOTE(k_neighbors=_k_eff, random_state=_seed, sampling_strategy=sampling_strategy)")
    else:
        lines.append("        sm = SMOTE(k_neighbors=_k_eff, random_state=_seed, sampling_strategy=sampling_strategy)")
    lines.append("        X_res, y_res = sm.fit_resample(X, y)")
    lines.append("        out_df = pd.concat([pd.DataFrame(X_res, columns=feat_cols, index=None),")
    lines.append("                           pd.Series(y_res, name=_target)], axis=1)")
    lines.append("    except Exception as e:")
    lines.append("        # Be forgiving: if SMOTE fails, fall back to original data")
    lines.append("        out_df = df.copy()")
    lines.append("        # Optional: annotate error in a side-channel if desired")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """Generate the Python body for the SMOTE node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: A list of code lines representing the body of the node.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_smote_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Input (single table)
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    # Transform + outputs
    lines.extend(_emit_smote_code(cfg))

    # Publish to context (default single output port '1')
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
    """Generate the code for a Jupyter notebook cell for the SMOTE node.

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
    """Handle the node processing and return the necessary imports and body lines.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming connections.
        outgoing: The outgoing connections.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines if the node can be handled; None otherwise.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
