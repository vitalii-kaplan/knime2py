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

from pathlib import Path
from typing import List, Optional

from .node_utils import (  # normalize_in_ports, collect_module_imports, split_out_imports
    collect_module_imports,
    normalize_in_ports,
    split_out_imports,
)
from .normalizer_utils import (
    NormalizerSettings,
    emit_normalize_code,
    parse_normalizer_settings,
)

# KNIME factory for Normalizer
FACTORY = "org.knime.base.node.preproc.normalize3.Normalizer3NodeFactory"

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    """Generate the necessary imports for the normalization process."""
    return ["import pandas as pd", "import math"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.normalize3.Normalizer3NodeFactory"
)

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
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

    lines.extend(emit_normalize_code(cfg))

    # Publish (default port '1')
    ports = out_ports or ["1"]
    port_map = {"1": "out_df", "2": "bundle"}
    for p in sorted({(p or '1') for p in ports}):
        target = port_map.get(p, "out_df")
        lines.append(f"context['{node_id}:{p}'] = {target}")

    return lines



def get_name() -> str:
    """Return name of the node in KNIME workflow."""
    return "Normalizer"


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
