#!/usr/bin/env python3

"""
Model Writer node handler.

Serializes arbitrary Python objects (models) to disk using pickle, mirroring KNIME's
Model Writer node behavior.
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
    normalize_in_ports,
    resolve_reader_path,
    split_out_imports,
)

FACTORY = "org.knime.base.node.io.filehandling.model.writer.ModelWriterNodeFactory"
HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.io.filehandling.model.writer.ModelWriterNodeFactory"
)


@dataclass
class ModelWriterSettings:
    path: Optional[str] = None
    create_dirs: bool = False


def parse_model_writer_settings(node_dir: Optional[Path]) -> ModelWriterSettings:
    if not node_dir:
        return ModelWriterSettings()
    sp = node_dir / "settings.xml"
    if not sp.exists():
        return ModelWriterSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()

    try:
        resolved = resolve_reader_path(root, node_dir)
        path = str(resolved) if resolved else None
    except Exception:
        path = None
    if not path:
        path = first(root, ".//*[local-name()='config' and @key='path']/*[local-name()='entry' and @key='path']/@value")

    create_dirs = first(root, ".//*[local-name()='entry' and @key='create_missing_folders']/@value")

    return ModelWriterSettings(
        path=path,
        create_dirs=str(create_dirs or "").strip().lower() == "true",
    )


def generate_imports(settings: ModelWriterSettings) -> List[str]:
    return ["from pathlib import Path", "import pickle"]


def generate_py_body(
    node_id: str,
    settings: ModelWriterSettings,
    in_ports: List[tuple[str, str]],
) -> List[str]:
    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    src_id, in_port = normalize_in_ports(in_ports)[0]
    lines.append(f"model_obj = context['{src_id}:{in_port}']")

    if settings.path:
        lines.append(f"out_path = Path(r\"{settings.path}\")")
    else:
        lines.append("# Model output path missing; adjust manually.")
        lines.append("out_path = Path('model_output.pkl')")

    if settings.create_dirs:
        lines.append("out_path.parent.mkdir(parents=True, exist_ok=True)")
    else:
        lines.append("out_path.parent.mkdir(parents=True, exist_ok=True)")

    lines.append("with out_path.open('wb') as fh:")
    lines.append("    pickle.dump(model_obj, fh)")
    lines.append(f"context['{node_id}:1'] = str(out_path)")
    return lines



def get_name() -> str:
    """Return human-readable handler name."""
    return "Model Writer"


def handle(ntype, nid, npath, incoming, outgoing):
    in_ports = [(src, str(getattr(edge, "source_port", "") or "1")) for src, edge in (incoming or [])]
    settings = parse_model_writer_settings(Path(npath) if npath else None)
    node_lines = generate_py_body(nid, settings, in_ports)
    explicit_imports = collect_module_imports(lambda: generate_imports(settings))
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
