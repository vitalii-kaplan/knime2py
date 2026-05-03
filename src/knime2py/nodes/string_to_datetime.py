#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, iter_entries, normalize_in_ports, split_out_imports


FACTORY = "org.knime.time.node.convert.stringtodatetime.StringToDateTimeNodeFactory2"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.time/latest/"
    "org.knime.time.node.convert.stringtodatetime.StringToDateTimeNodeFactory2"
)


@dataclass
class StringToDateTimeSettings:
    columns: List[str] = field(default_factory=list)
    knime_format: str = ""
    pandas_format: Optional[str] = None
    temporal_type: str = "DATE_TIME"
    on_error: str = "SET_MISSING"
    append_or_replace: str = "REPLACE"
    output_suffix: str = " (Date&time)"


def _indexed_values(parent: Optional[ET._Element]) -> List[str]:
    if parent is None:
        return []
    values: list[tuple[int, str]] = []
    for key, value in iter_entries(parent):
        if key.isdigit() and value is not None:
            values.append((int(key), value))
    return [value for _, value in sorted(values)]


def _knime_to_strptime(fmt: str) -> Optional[str]:
    if not fmt:
        return None
    replacements = [
        ("yyyy", "%Y"),
        ("YYYY", "%Y"),
        ("yy", "%y"),
        ("MM", "%m"),
        ("dd", "%d"),
        ("HH", "%H"),
        ("hh", "%I"),
        ("mm", "%M"),
        ("ss", "%S"),
        ("SSS", "%f"),
    ]
    out = fmt
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


class StringToDateTime:
    @staticmethod
    def parse(node_dir: Optional[Path]) -> StringToDateTimeSettings:
        if not node_dir:
            return StringToDateTimeSettings()
        settings_path = node_dir / "settings.xml"
        if not settings_path.exists():
            return StringToDateTimeSettings()

        root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
        model = first_el(root, ".//*[local-name()='config' and @key='model']")
        if model is None:
            return StringToDateTimeSettings()

        columns = _indexed_values(
            first_el(
                model,
                "./*[local-name()='config' and @key='columnFilter']"
                "/*[local-name()='config' and @key='manualFilter']"
                "/*[local-name()='config' and @key='manuallySelected']",
            )
        )
        knime_format = (
            first(
                model,
                "./*[local-name()='config' and @key='format']"
                "/*[local-name()='entry' and @key='format']/@value",
            )
            or ""
        )

        return StringToDateTimeSettings(
            columns=columns,
            knime_format=knime_format,
            pandas_format=_knime_to_strptime(knime_format),
            temporal_type=(
                first(
                    model,
                    "./*[local-name()='config' and @key='format']"
                    "/*[local-name()='entry' and @key='temporalType']/@value",
                )
                or "DATE_TIME"
            ),
            on_error=first(model, "./*[local-name()='entry' and @key='onError']/@value") or "SET_MISSING",
            append_or_replace=first(model, "./*[local-name()='entry' and @key='appendOrReplace']/@value") or "REPLACE",
            output_suffix=first(model, "./*[local-name()='entry' and @key='outputColumnSuffix']/@value")
            or " (Date&time)",
        )

    @staticmethod
    def emit(settings: StringToDateTimeSettings) -> List[str]:
        return [
            f"_datetime_columns = {repr(settings.columns)}",
            f"_datetime_format = {repr(settings.pandas_format)}",
            f"_temporal_type = {repr(settings.temporal_type)}",
            f"_on_error = {repr(settings.on_error)}",
            f"_append_or_replace = {repr(settings.append_or_replace)}",
            f"_output_suffix = {repr(settings.output_suffix)}",
            "out_df = df.copy()",
            "_errors = 'coerce' if str(_on_error).upper() == 'SET_MISSING' else 'raise'",
            "for _col in _datetime_columns:",
            "    if _col not in out_df.columns:",
            "        continue",
            "    _converted = pd.to_datetime(out_df[_col], format=_datetime_format, errors=_errors)",
            "    if str(_temporal_type).upper() == 'DATE':",
            "        _converted = _converted.dt.normalize()",
            "    elif str(_temporal_type).upper() == 'TIME':",
            "        _converted = _converted.dt.time",
            "    _target = _col if str(_append_or_replace).upper() == 'REPLACE' else f'{_col}{_output_suffix}'",
            "    out_df[_target] = _converted",
        ]


def parse_string_to_datetime_settings(node_dir: Optional[Path]) -> StringToDateTimeSettings:
    return StringToDateTime.parse(node_dir)


def generate_imports() -> List[str]:
    return ["import pandas as pd"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_string_to_datetime_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']  # input table",
    ]
    lines.extend(StringToDateTime.emit(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "String to Date&Time"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
