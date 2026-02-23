#!/usr/bin/env python3

"""
Normalizer (PMML) node handler.

This node mirrors the behavior of the standard Normalizer but emits a PMML document
describing the applied transformations on its second port.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .node_utils import collect_module_imports, normalize_in_ports, split_out_imports
from .normalizer_utils import (
    NormalizerSettings,
    emit_normalize_code,
    parse_normalizer_settings,
)
from .pmml_utils import emit_data_dictionary_helper

FACTORY = "org.knime.base.node.preproc.pmml.normalize.NormalizerPMMLNodeFactory2"
DATA_DICT_HELPER_LINES = emit_data_dictionary_helper("_norm_dtype_key", "_collect_norm_data_dictionary")

def generate_imports() -> List[str]:
    return [
        "import pandas as pd",
        "import math",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.pmml.normalize.NormalizerPMMLNodeFactory2"
)


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_normalizer_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    src_id, in_port = normalize_in_ports(in_ports)[0]
    lines.append(f"df = context['{src_id}:{in_port}']")

    lines.extend(DATA_DICT_HELPER_LINES)
    lines.extend(emit_normalize_code(cfg))
    lines.append("data_dictionary = _collect_norm_data_dictionary(df)")
    lines.append("pmml_model = {")
    lines.append("    'model_type': 'normalizer',")
    lines.append("    'version': '4.2',")
    lines.append("    'application': {'name': 'knime2py', 'version': '1.0'},")
    lines.append("    'mode': bundle.get('mode', 'MINMAX'),")
    lines.append("    'new_min': bundle.get('new_min', 0.0),")
    lines.append("    'new_max': bundle.get('new_max', 1.0),")
    lines.append("    'columns': list(bundle.get('columns', [])),")
    lines.append("    'stats': dict(bundle.get('stats', {})),")
    lines.append("    'data_dictionary': data_dictionary,")
    lines.append("}")

    ports = out_ports or ["1", "2"]
    port_map = {"1": "out_df", "2": "pmml_model"}
    for p in sorted({(p or '1') for p in ports}):
        target = port_map.get(p, "out_df")
        lines.append(f"context['{node_id}:{p}'] = {target}")

    return lines



def get_name() -> str:
    """Return human-readable handler name."""
    return "Normalizer PMML"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src, str(getattr(edge, "source_port", "") or "1")) for src, edge in (incoming or [])]
    out_ports = [str(getattr(edge, "source_port", "") or "1") for _, edge in (outgoing or [])] or ["1", "2"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
