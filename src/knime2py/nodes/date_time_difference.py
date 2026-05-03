#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, normalize_in_ports, split_out_imports


FACTORY = "org.knime.time.node.calculate.datetimedifference.DateTimeDifferenceNodeFactory2"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.time/latest/"
    "org.knime.time.node.calculate.datetimedifference.DateTimeDifferenceNodeFactory2"
)


@dataclass
class DateTimeDifferenceSettings:
    first_column: str = ""
    second_value_type: str = "COLUMN"
    second_column: str = ""
    local_datetime_fixed: str = ""
    mode: str = "SECOND_MINUS_FIRST"
    output_type: str = "DURATION_OR_PERIOD"
    granularity: str = "YEAR"
    output_number_type: str = "NO_DECIMALS"
    output_column_name: str = "Date&Time Difference"


class DateTimeDifference:
    @staticmethod
    def parse(node_dir: Optional[Path]) -> DateTimeDifferenceSettings:
        if not node_dir:
            return DateTimeDifferenceSettings()
        settings_path = node_dir / "settings.xml"
        if not settings_path.exists():
            return DateTimeDifferenceSettings()

        root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
        model = first_el(root, ".//*[local-name()='config' and @key='model']")
        if model is None:
            return DateTimeDifferenceSettings()

        return DateTimeDifferenceSettings(
            first_column=first(model, "./*[local-name()='entry' and @key='firstColumnSelection']/@value") or "",
            second_value_type=first(model, "./*[local-name()='entry' and @key='secondDateTimeValueType']/@value")
            or "COLUMN",
            second_column=first(model, "./*[local-name()='entry' and @key='secondColumnSelection']/@value") or "",
            local_datetime_fixed=first(model, "./*[local-name()='entry' and @key='localDateTimeFixed']/@value") or "",
            mode=first(model, "./*[local-name()='entry' and @key='mode']/@value") or "SECOND_MINUS_FIRST",
            output_type=first(model, "./*[local-name()='entry' and @key='outputType']/@value") or "DURATION_OR_PERIOD",
            granularity=first(model, "./*[local-name()='entry' and @key='granularity']/@value") or "YEAR",
            output_number_type=first(model, "./*[local-name()='entry' and @key='outputNumberType']/@value")
            or "NO_DECIMALS",
            output_column_name=first(model, "./*[local-name()='entry' and @key='outputColumnName']/@value")
            or "Date&Time Difference",
        )

    @staticmethod
    def emit(settings: DateTimeDifferenceSettings) -> List[str]:
        return [
            f"_first_col = {repr(settings.first_column)}",
            f"_second_value_type = {repr(settings.second_value_type)}",
            f"_second_col = {repr(settings.second_column)}",
            f"_fixed_datetime = {repr(settings.local_datetime_fixed)}",
            f"_mode = {repr(settings.mode)}",
            f"_output_col = {repr(settings.output_column_name)}",
            "out_df = df.copy()",
            "if _first_col in out_df.columns:",
            "    _first = pd.to_datetime(out_df[_first_col], errors='coerce')",
            "else:",
            "    _first = pd.Series(pd.NaT, index=out_df.index)",
            "if str(_second_value_type).upper() == 'COLUMN' and _second_col in out_df.columns:",
            "    _second = pd.to_datetime(out_df[_second_col], errors='coerce')",
            "else:",
            "    _second = pd.Series(pd.to_datetime(_fixed_datetime, errors='coerce'), index=out_df.index)",
            "if str(_mode).upper() == 'FIRST_MINUS_SECOND':",
            "    out_df[_output_col] = _first - _second",
            "else:",
            "    out_df[_output_col] = _second - _first",
        ]


def parse_date_time_difference_settings(node_dir: Optional[Path]) -> DateTimeDifferenceSettings:
    return DateTimeDifference.parse(node_dir)


def generate_imports() -> List[str]:
    return ["import pandas as pd"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_date_time_difference_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']  # input table",
    ]
    lines.extend(DateTimeDifference.emit(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "Date&Time Difference"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
