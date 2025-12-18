#!/usr/bin/env python3

"""
PMML Writer node handler.

Writes a PMML string/bytes from the context to disk using the path defined in settings.xml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import (
    collect_module_imports,
    first,
    first_el,
    normalize_in_ports,
    resolve_reader_path,
    split_out_imports,
)

FACTORY = "org.knime.base.node.io.filehandling.pmml.writer.PMMLWriterNodeFactory2"
HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.io.filehandling.pmml.writer.PMMLWriterNodeFactory2"
)


@dataclass
class PMMLWriterSettings:
    path: Optional[str] = None
    create_dirs: bool = False
    validate: bool = False


def parse_pmml_writer_settings(node_dir: Optional[Path]) -> PMMLWriterSettings:
    if not node_dir:
        return PMMLWriterSettings()
    sp = node_dir / "settings.xml"
    if not sp.exists():
        return PMMLWriterSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()

    path: Optional[str]
    try:
        resolved = resolve_reader_path(root, node_dir)
        path = str(resolved) if resolved else None
    except Exception:
        path = None
    if not path:
        path = first(root, ".//*[local-name()='config' and @key='path']/*[local-name()='entry' and @key='path']/@value")

    create_dirs = first(root, ".//*[local-name()='entry' and @key='create_missing_folders']/@value")
    validate = first(root, ".//*[local-name()='entry' and @key='validate_PMML']/@value")

    return PMMLWriterSettings(
        path=path,
        create_dirs=str(create_dirs or "").strip().lower() == "true",
        validate=str(validate or "").strip().lower() == "true",
    )


def generate_imports(settings: PMMLWriterSettings) -> List[str]:
    imports = ["from pathlib import Path"]
    if settings.validate:
        imports.append("from xml.etree import ElementTree as ET")
    return imports


def generate_py_body(
    node_id: str,
    settings: PMMLWriterSettings,
    in_ports: List[tuple[str, str]],
) -> List[str]:

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    src_id, in_port = normalize_in_ports(in_ports)[0]
    lines.append(f"pmml_obj = context['{src_id}:{in_port}']")

    if settings.path:
        lines.append(f"out_path = Path(r\"{settings.path}\")")
    else:
        lines.append("# PMML output path missing; please adjust manually.")
        lines.append("out_path = Path('pmml_output.pmml')")

    if settings.create_dirs:
        lines.append("out_path.parent.mkdir(parents=True, exist_ok=True)")
    else:
        lines.append("out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists")

    lines.append("if isinstance(pmml_obj, (bytes, bytearray)):")
    lines.append("    pmml_text = pmml_obj.decode('utf-8')")
    lines.append("else:")
    lines.append("    pmml_text = str(pmml_obj)")

    if settings.validate:
        lines.append("try:")
        lines.append("    ET.fromstring(pmml_text)")
        lines.append("except ET.ParseError as exc:")
        lines.append("    raise ValueError('Invalid PMML detected by PMML Writer') from exc")

    lines.append("pmml_bytes = pmml_text.encode('utf-8')")
    lines.append("with out_path.open('wb') as fh:")
    lines.append("    fh.write(pmml_bytes)")
    lines.append(f"context['{node_id}:1'] = str(out_path)")

    return lines


def handle(ntype, nid, npath, incoming, outgoing):
    in_ports = [(src_id, str(getattr(edge, "source_port", "") or "1")) for src_id, edge in (incoming or [])]
    settings = parse_pmml_writer_settings(Path(npath) if npath else None)
    node_lines = generate_py_body(nid, settings, in_ports)
    explicit_imports = collect_module_imports(lambda: generate_imports(settings))
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
