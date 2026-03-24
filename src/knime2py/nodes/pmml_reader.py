#!/usr/bin/env python3

"""
PMML Reader node handler.

Loads a PMML document from disk (resolving KNIME-style relative paths) and
publishes the XML text to the outgoing context port(s).
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
    resolve_reader_path,
    split_out_imports,
)

FACTORY = "org.knime.base.node.io.filehandling.pmml.reader.PMMLReaderNodeFactory3"
HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.io.filehandling.pmml.reader.PMMLReaderNodeFactory3"
)


@dataclass
class PMMLReaderSettings:
    path: Optional[str] = None


def parse_pmml_reader_settings(node_dir: Optional[Path]) -> PMMLReaderSettings:
    if not node_dir:
        return PMMLReaderSettings()

    settings_file = node_dir / "settings.xml"
    if not settings_file.exists():
        return PMMLReaderSettings()

    root = ET.parse(str(settings_file), parser=XML_PARSER).getroot()

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

    return PMMLReaderSettings(path=resolved)


def generate_imports() -> List[str]:
    return [
        "from pathlib import Path",
        "import xml.etree.ElementTree as ET",
    ]


def generate_py_body(
    node_id: str,
    settings: PMMLReaderSettings,
    out_ports: List[str],
) -> List[str]:
    ports = out_ports or ["1"]
    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    if settings.path:
        lines.append(f"pmml_path = Path(r\"{settings.path}\")")
    else:
        lines.append("# PMML path missing in settings; please update manually.")
        lines.append("pmml_path = Path('path/to/model.pmml')")

    lines.append("if not pmml_path.exists():")
    lines.append("    raise FileNotFoundError(f\"PMML file not found: {pmml_path}\")")
    lines.append("pmml_text = pmml_path.read_text(encoding='utf-8')")
    lines.append("")
    lines.append("def _bundle_from_pmml(pmml_text):")
    lines.append("    try:")
    lines.append("        root = ET.fromstring(pmml_text)")
    lines.append("    except ET.ParseError as exc:")
    lines.append("        raise ValueError('Failed to parse PMML document') from exc")
    lines.append("    version = root.get('version') or '4.2'")
    lines.append("    header_app = root.find('.//{*}Header/{*}Application')")
    lines.append("    app_name = 'knime2py'")
    lines.append("    app_version = '1.0'")
    lines.append("    if header_app is not None:")
    lines.append("        app_name = header_app.get('name') or app_name")
    lines.append("        app_version = header_app.get('version') or app_version")
    lines.append("    data_types = {}")
    lines.append("    for field in root.findall('.//{*}DataDictionary/{*}DataField'):")
    lines.append("        name = field.get('name')")
    lines.append("        if not name:")
    lines.append("            continue")
    lines.append("        data_types[name] = (field.get('dataType') or '').lower()")
    lines.append("    derived_fields = root.findall('.//{*}TransformationDictionary/{*}DerivedField')")
    lines.append("    if not derived_fields:")
    lines.append("        raise ValueError('No TransformationDictionary entries found in PMML')")
    lines.append("    norm_columns = []")
    lines.append("    norm_stats = {}")
    lines.append("    norm_new_min = None")
    lines.append("    norm_new_max = None")
    lines.append("    column_strategies = []")
    lines.append("    for derived in derived_fields:")
    lines.append("        norm = derived.find('.//{*}NormContinuous')")
    lines.append("        if norm is not None:")
    lines.append("            column = norm.get('field') or norm.get('origField')")
    lines.append("            linear = norm.findall('.//{*}LinearNorm')")
    lines.append("            if not column or len(linear) < 2:")
    lines.append("                continue")
    lines.append("            try:")
    lines.append("                mn = float(linear[0].get('orig'))")
    lines.append("                new_min = float(linear[0].get('norm'))")
    lines.append("                mx = float(linear[1].get('orig'))")
    lines.append("                new_max = float(linear[1].get('norm'))")
    lines.append("            except (TypeError, ValueError):")
    lines.append("                continue")
    lines.append("            if norm_new_min is None:")
    lines.append("                norm_new_min = new_min")
    lines.append("                norm_new_max = new_max")
    lines.append("            norm_columns.append(column)")
    lines.append("            norm_stats[column] = {'min': mn, 'max': mx}")
    lines.append("            continue")
    lines.append("        field_refs = derived.findall('.//{*}FieldRef')")
    lines.append("        const = derived.find('.//{*}Constant')")
    lines.append("        if not field_refs or const is None:")
    lines.append("            continue")
    lines.append("        column = field_refs[0].get('field')")
    lines.append("        if not column:")
    lines.append("            continue")
    lines.append("        dtype_raw = data_types.get(column, '')")
    lines.append("        dtype_key = 'string'")
    lines.append("        dr = dtype_raw.lower()")
    lines.append("        if 'int' in dr:")
    lines.append("            dtype_key = 'int'")
    lines.append("        elif dr in {'double', 'float'}:")
    lines.append("            dtype_key = 'float'")
    lines.append("        elif 'bool' in dr:")
    lines.append("            dtype_key = 'boolean'")
    lines.append("        column_strategies.append({")
    lines.append("            'column': column,")
    lines.append("            'dtype': dtype_key,")
    lines.append("            'strategy': 'fixed',")
    lines.append("            'value': const.text,")
    lines.append("        })")
    lines.append("    if norm_columns:")
    lines.append("        return {")
    lines.append("            'model_type': 'normalizer',")
    lines.append("            'version': version,")
    lines.append("            'application': {'name': app_name, 'version': app_version},")
    lines.append("            'mode': 'MINMAX',")
    lines.append("            'new_min': norm_new_min if norm_new_min is not None else 0.0,")
    lines.append("            'new_max': norm_new_max if norm_new_max is not None else 1.0,")
    lines.append("            'columns': norm_columns,")
    lines.append("            'stats': norm_stats,")
    lines.append("        }")
    lines.append("    return {'strategies': [], 'column_strategies': column_strategies}")
    lines.append("")
    lines.append("model_obj = _bundle_from_pmml(pmml_text)")

    for port in ports:
        port_id = port or "1"
        lines.append(f"context['{node_id}:{port_id}'] = model_obj")

    return lines



def get_name() -> str:
    """Return name of the node in KNIME workflow."""
    return "PMML Reader"


def handle(ntype, nid, npath, incoming, outgoing):
    out_ports = [str(getattr(edge, "source_port", "") or "1") for _, edge in outgoing]
    node_dir = Path(npath) if npath else None
    settings = parse_pmml_reader_settings(node_dir)
    node_lines = generate_py_body(nid, settings, out_ports)

    found_imports, body = split_out_imports(node_lines)
    explicit_imports = collect_module_imports(generate_imports)
    imports = sorted(set(found_imports) | set(explicit_imports))
    return imports, body
