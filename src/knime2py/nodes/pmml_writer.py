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

PMML_HELPER_LINES = [
    "def _pmml_from_missing_value_bundle(bundle):",
    "    version = str(bundle.get('version') or '4.2')",
    "    ns = 'http://www.dmg.org/PMML-4_2'",
    "    root = ET.Element('PMML', version=version, xmlns=ns)",
    "    header = ET.SubElement(root, 'Header')",
    "    app = bundle.get('application') or {}",
    "    ET.SubElement(header, 'Application', name=str(app.get('name', 'knime2py')), version=str(app.get('version', '1.0')))",
    "    data_entries = bundle.get('data_dictionary') or []",
    "    data_dict = ET.SubElement(root, 'DataDictionary', numberOfFields=str(len(data_entries)))",
    "    for entry in data_entries:",
    "        name = str(entry.get('name', 'field'))",
    "        optype = str(entry.get('optype', 'continuous'))",
    "        dtype = str(entry.get('dataType', 'string'))",
    "        field = ET.SubElement(data_dict, 'DataField', name=name, optype=optype, dataType=dtype)",
    "        values = entry.get('values') or []",
    "        for val in values[:256]:",
    "            ET.SubElement(field, 'Value', value=str(val))",
    "        interval = entry.get('interval')",
    "        if isinstance(interval, (list, tuple)) and len(interval) == 2:",
    "            ET.SubElement(field, 'Interval', closure='closedClosed', leftMargin=str(interval[0]), rightMargin=str(interval[1]))",
    "    trans_dict = ET.SubElement(root, 'TransformationDictionary')",
    "    for info in bundle.get('transformations') or []:",
    "        column = str(info.get('column', 'field'))",
    "        derived_name = str(info.get('derived_name') or f\"{column}*\")",
    "        optype = str(info.get('optype', 'continuous'))",
    "        dtype = str(info.get('const_dtype', 'string'))",
    "        derived = ET.SubElement(trans_dict, 'DerivedField', name=derived_name, displayName=column, optype=optype, dataType=dtype)",
    "        apply_if = ET.SubElement(derived, 'Apply', function='if')",
    "        apply_missing = ET.SubElement(apply_if, 'Apply', function='isMissing')",
    "        ET.SubElement(apply_missing, 'FieldRef', field=column)",
    "        const_el = ET.SubElement(apply_if, 'Constant', dataType=dtype)",
    "        const_el.text = str(info.get('const', ''))",
    "        ET.SubElement(apply_if, 'FieldRef', field=column)",
    "    return ET.tostring(root, encoding='utf-8', xml_declaration=True).decode('utf-8')",
    "",
    "def _pmml_from_normalizer_bundle(bundle):",
    "    version = str(bundle.get('version') or '4.2')",
    "    ns = 'http://www.dmg.org/PMML-4_2'",
    "    root = ET.Element('PMML', version=version, xmlns=ns)",
    "    header = ET.SubElement(root, 'Header')",
    "    app = bundle.get('application') or {}",
    "    ET.SubElement(header, 'Application', name=str(app.get('name', 'knime2py')), version=str(app.get('version', '1.0')))",
    "    cols = [c for c in (bundle.get('columns') or []) if c]",
    "    stats = bundle.get('stats') or {}",
    "    mode = str(bundle.get('mode', 'MINMAX')).upper()",
    "    new_min = bundle.get('new_min', 0.0)",
    "    new_max = bundle.get('new_max', 1.0)",
    "    dd_entries = bundle.get('data_dictionary')",
    "    if dd_entries:",
    "        data_dict = ET.SubElement(root, 'DataDictionary', numberOfFields=str(len(dd_entries)))",
    "        for entry in dd_entries:",
    "            name = str(entry.get('name', 'field'))",
    "            optype = str(entry.get('optype', 'continuous'))",
    "            dtype = str(entry.get('dataType', 'string'))",
    "            field = ET.SubElement(data_dict, 'DataField', name=name, optype=optype, dataType=dtype)",
    "            values = entry.get('values') or []",
    "            for val in values[:256]:",
    "                ET.SubElement(field, 'Value', value=str(val))",
    "            interval = entry.get('interval')",
    "            if isinstance(interval, (list, tuple)) and len(interval) == 2:",
    "                ET.SubElement(",
    "                    field,",
    "                    'Interval',",
    "                    closure=str(entry.get('intervalClosure', 'closedClosed')),",
    "                    leftMargin=str(interval[0]),",
    "                    rightMargin=str(interval[1]),",
    "                )",
    "    else:",
    "        data_dict = ET.SubElement(root, 'DataDictionary', numberOfFields=str(len(cols)))",
    "        for col in cols:",
    "            field = ET.SubElement(",
    "                data_dict,",
    "                'DataField',",
    "                name=str(col),",
    "                optype='continuous',",
    "                dataType='double',",
    "            )",
    "            info = stats.get(col) or {}",
    "            if 'min' in info and 'max' in info and info.get('min') is not None and info.get('max') is not None:",
    "                ET.SubElement(",
    "                    field,",
    "                    'Interval',",
    "                    closure='closedClosed',",
    "                    leftMargin=str(info.get('min')),",
    "                    rightMargin=str(info.get('max')),",
    "                )",
    "            elif 'mean' in info and 'std' in info and info.get('std') not in (None, 0):",
    "                mu = float(info.get('mean') or 0.0)",
    "                sd = abs(float(info.get('std')))",
    "                left = mu - 3 * sd",
    "                right = mu + 3 * sd",
    "                ET.SubElement(",
    "                    field,",
    "                    'Interval',",
    "                    closure='closedClosed',",
    "                    leftMargin=str(left),",
    "                    rightMargin=str(right),",
    "                )",
    "    if not cols:",
    "        return ET.tostring(root, encoding='utf-8', xml_declaration=True).decode('utf-8')",
    "    trans = ET.SubElement(root, 'TransformationDictionary')",
    "    col_count = len(cols)",
    "    for col in cols:",
    "        info = stats.get(col) or {}",
    "        summary = None",
    "        if mode == 'MINMAX':",
    "            mn = info.get('min')",
    "            mx = info.get('max')",
    "            if mn is None or mx is None:",
    "                continue",
    "            derived = ET.SubElement(",
    "                trans,",
    "                'DerivedField',",
    "                name=f\"{col}_norm\",",
    "                optype='continuous',",
    "                dataType='double',",
    "            )",
    "            summary = f\"Min/Max ({new_min}, {new_max}) normalization on {col_count} column(s)\"",
    "            norm = ET.SubElement(derived, 'NormContinuous', field=str(col))",
    "            ET.SubElement(norm, 'LinearNorm', orig=str(mn), norm=str(new_min))",
    "            ET.SubElement(norm, 'LinearNorm', orig=str(mx), norm=str(new_max))",
    "        elif mode == 'ZSCORE':",
    "            mu = info.get('mean')",
    "            sd = info.get('std')",
    "            if sd in (None, 0):",
    "                continue",
    "            derived = ET.SubElement(",
    "                trans,",
    "                'DerivedField',",
    "                name=f\"{col}_z\",",
    "                optype='continuous',",
    "                dataType='double',",
    "            )",
    "            summary = f\"Z Score normalization on {col_count} column(s)\"",
    "            apply_div = ET.SubElement(derived, 'Apply', function='/')",
    "            apply_sub = ET.SubElement(apply_div, 'Apply', function='-')",
    "            ET.SubElement(apply_sub, 'FieldRef', field=str(col))",
    "            const_mu = ET.SubElement(apply_sub, 'Constant', dataType='double')",
    "            const_mu.text = '0.0' if mu is None else str(mu)",
    "            const_sd = ET.SubElement(apply_div, 'Constant', dataType='double')",
    "            const_sd.text = str(sd)",
    "        elif mode == 'DECIMALSCALING':",
    "            scale = info.get('scale')",
    "            try:",
    "                scale_int = int(scale)",
    "            except Exception:",
    "                scale_int = None",
    "            if scale_int in (None, 0):",
    "                continue",
    "            derived = ET.SubElement(",
    "                trans,",
    "                'DerivedField',",
    "                name=f\"{col}_dec\",",
    "                optype='continuous',",
    "                dataType='double',",
    "            )",
    "            summary = f\"Decimal scaling normalization on {col_count} column(s)\"",
    "            apply_div = ET.SubElement(derived, 'Apply', function='/')",
    "            ET.SubElement(apply_div, 'FieldRef', field=str(col))",
    "            const_scale = ET.SubElement(apply_div, 'Constant', dataType='double')",
    "            const_scale.text = str(float(10 ** scale_int))",
    "        else:",
    "            continue",
    "        if summary:",
    "            ET.SubElement(derived, 'Extension', name='summary', extender='knime2py', value=summary)",
    "    return ET.tostring(root, encoding='utf-8', xml_declaration=True).decode('utf-8')",
    "",
    "def _pmml_pretty_print(xml_text):",
    "    try:",
    "        minidom = __import__('xml.dom.minidom', fromlist=['minidom'])",
    "    except Exception:",
    "        return xml_text",
    "    try:",
    "        parsed = minidom.parseString(xml_text.encode('utf-8'))",
    "    except Exception:",
    "        return xml_text",
    "    pretty = parsed.toprettyxml(indent='  ', encoding='UTF-8')",
    "    lines = [line for line in pretty.decode('utf-8').splitlines() if line.strip()]",
    "    return '\\n'.join(lines)",
]


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
    imports = ["from pathlib import Path", "from xml.etree import ElementTree as ET"]
    return imports


def generate_py_body(
    node_id: str,
    settings: PMMLWriterSettings,
    in_ports: List[tuple[str, str]],
) -> List[str]:

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.extend(PMML_HELPER_LINES)

    src_id, in_port = normalize_in_ports(in_ports)[0]
    lines.append(f"pmml_obj = context['{src_id}:{in_port}']")
    lines.append("pmml_text = None")
    lines.append("if isinstance(pmml_obj, dict):")
    lines.append("    model_type = str(pmml_obj.get('model_type', '')).lower()")
    lines.append("    if model_type == 'missing_value':")
    lines.append("        pmml_text = _pmml_from_missing_value_bundle(pmml_obj)")
    lines.append("    elif model_type == 'normalizer':")
    lines.append("        pmml_text = _pmml_from_normalizer_bundle(pmml_obj)")
    lines.append("    else:")
    lines.append("        pmml_text = str(pmml_obj)")
    lines.append("    pmml_text = _pmml_pretty_print(pmml_text)")
    lines.append("elif isinstance(pmml_obj, (bytes, bytearray)):")
    lines.append("    pmml_text = pmml_obj.decode('utf-8')")
    lines.append("else:")
    lines.append("    pmml_text = str(pmml_obj)")
    lines.append("pmml_text = pmml_text or ''")

    if settings.path:
        lines.append(f"out_path = Path(r\"{settings.path}\")")
    else:
        lines.append("# PMML output path missing; please adjust manually.")
        lines.append("out_path = Path('pmml_output.pmml')")

    if settings.create_dirs:
        lines.append("out_path.parent.mkdir(parents=True, exist_ok=True)")
    else:
        lines.append("out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists")

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
