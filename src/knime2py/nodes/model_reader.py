#!/usr/bin/env python3

"""
Model Reader node handler.

Loads a pickle-serialized model (written by our Model Writer handler) and publishes
the object to the outgoing port.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import (
    collect_module_imports,
    first,
    resolve_reader_path,
    split_out_imports,
)

FACTORY = "org.knime.base.node.io.filehandling.model.reader.ModelReaderNodeFactory"
HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.io.filehandling.model.reader.ModelReaderNodeFactory"
)


@dataclass
class ModelReaderSettings:
    path: Optional[str] = None


def parse_model_reader_settings(node_dir: Optional[Path]) -> ModelReaderSettings:
    if not node_dir:
        return ModelReaderSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return ModelReaderSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    resolved: Optional[str] = None
    try:
        resolved = resolve_reader_path(root, node_dir)
    except Exception:
        resolved = None

    if not resolved:
        resolved = first(
            root,
            ".//*[local-name()='config' and @key='path']"
            "/*[local-name()='entry' and @key='path']/@value",
        )

    return ModelReaderSettings(path=resolved)


def generate_imports() -> List[str]:
    return ["from pathlib import Path", "import pickle"]


def generate_py_body(
    node_id: str,
    settings: ModelReaderSettings,
    out_ports: List[str],
) -> List[str]:
    ports = out_ports or ["1"]
    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    if settings.path:
        lines.append(f"model_path = Path(r\"{settings.path}\")")
    else:
        lines.append("# Model path missing in settings.xml; adjust manually.")
        lines.append("model_path = Path('model.pkl')")

    lines.append("if not model_path.exists():")
    lines.append("    raise FileNotFoundError(f\"Model file not found: {model_path}\")")
    lines.append("with model_path.open('rb') as fh:")
    lines.append("    model_obj = pickle.load(fh)")

    for port in sorted({(p or '1') for p in ports}):
        lines.append(f"context['{node_id}:{port}'] = model_obj")

    return lines



def get_name() -> str:
    """Return name of the node in KNIME workflow."""
    return "Model Reader"


def handle(ntype, nid, npath, incoming, outgoing):
    out_ports = [str(getattr(edge, "source_port", "") or "1") for _, edge in (outgoing or [])]
    settings = parse_model_reader_settings(Path(npath) if npath else None)
    node_lines = generate_py_body(nid, settings, out_ports)
    found_imports, body = split_out_imports(node_lines)
    explicit_imports = collect_module_imports(generate_imports)
    imports = sorted(set(found_imports) | set(explicit_imports))
    return imports, body
