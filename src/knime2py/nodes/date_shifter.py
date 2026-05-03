#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, normalize_in_ports, split_out_imports


FACTORY = "org.knime.time.node.manipulate.datetimeshift.DateShiftNodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.time/latest/"
    "org.knime.time.node.manipulate.datetimeshift.DateShiftNodeFactory"
)


@dataclass
class DateShifterSettings:
    columns: List[str] = field(default_factory=list)
    shift_mode: str = "SHIFT_VALUE"
    shift_period_value: str = "P0D"
    period_column: str = ""
    numerical_column: str = ""
    granularity: str = "DAYS"
    replace_or_append: str = "REPLACE"
    output_column_suffix: str = " (Shifted)"


def _manual_selected_columns(model) -> List[str]:
    selected = first_el(
        model,
        "./*[local-name()='config' and @key='columnFilter']"
        "/*[local-name()='config' and @key='manualFilter']"
        "/*[local-name()='config' and @key='manuallySelected']",
    )
    if selected is None:
        return []
    entries = selected.xpath("./*[local-name()='entry' and @key!='array-size']/@value")
    return [str(value) for value in entries if str(value)]


class DateShifter:
    @staticmethod
    def parse(node_dir: Optional[Path]) -> DateShifterSettings:
        if not node_dir:
            return DateShifterSettings()
        settings_path = node_dir / "settings.xml"
        if not settings_path.exists():
            return DateShifterSettings()

        root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
        model = first_el(root, ".//*[local-name()='config' and @key='model']")
        if model is None:
            return DateShifterSettings()

        return DateShifterSettings(
            columns=_manual_selected_columns(model),
            shift_mode=first(model, "./*[local-name()='entry' and @key='shiftMode']/@value") or "SHIFT_VALUE",
            shift_period_value=first(model, "./*[local-name()='entry' and @key='shiftPeriodValue']/@value") or "P0D",
            period_column=first(model, "./*[local-name()='entry' and @key='periodColumn']/@value") or "",
            numerical_column=first(model, "./*[local-name()='entry' and @key='numericalColumn']/@value") or "",
            granularity=first(model, "./*[local-name()='entry' and @key='granularity']/@value") or "DAYS",
            replace_or_append=first(model, "./*[local-name()='entry' and @key='replaceOrAppend']/@value") or "REPLACE",
            output_column_suffix=first(model, "./*[local-name()='entry' and @key='outputColumnSuffix']/@value")
            or " (Shifted)",
        )

    @staticmethod
    def emit(settings: DateShifterSettings) -> List[str]:
        return [
            f"_shift_columns = {repr(settings.columns)}",
            f"_shift_mode = {repr(settings.shift_mode)}",
            f"_period_value = {repr(settings.shift_period_value)}",
            f"_period_column = {repr(settings.period_column)}",
            f"_numerical_column = {repr(settings.numerical_column)}",
            f"_granularity = {repr(settings.granularity)}",
            f"_replace_or_append = {repr(settings.replace_or_append)}",
            f"_suffix = {repr(settings.output_column_suffix)}",
            "out_df = df.copy()",
            "def _k2p_parse_period_offset(_value):",
            "    if pd.isna(_value):",
            "        return pd.NaT",
            "    _text = str(_value).strip()",
            "    _match = re.fullmatch(r'([+-])?P(?:(\\d+)Y)?(?:(\\d+)M)?(?:(\\d+)W)?(?:(\\d+)D)?', _text)",
            "    if not _match:",
            "        return pd.to_timedelta(_text, errors='coerce')",
            "    _sign = -1 if _match.group(1) == '-' else 1",
            "    _years = _sign * int(_match.group(2) or 0)",
            "    _months = _sign * int(_match.group(3) or 0)",
            "    _weeks = _sign * int(_match.group(4) or 0)",
            "    _days = _sign * int(_match.group(5) or 0)",
            "    return pd.DateOffset(years=_years, months=_months, weeks=_weeks, days=_days)",
            "def _k2p_numeric_date_delta(_series):",
            "    _unit = str(_granularity).strip().lower()",
            "    if _unit in {'years', 'year'}:",
            "        return _series.map(lambda _v: pd.DateOffset(years=int(_v)) if pd.notna(_v) else pd.NaT)",
            "    if _unit in {'months', 'month'}:",
            "        return _series.map(lambda _v: pd.DateOffset(months=int(_v)) if pd.notna(_v) else pd.NaT)",
            "    _unit_map = {'weeks': 'W', 'week': 'W', 'days': 'D', 'day': 'D'}",
            "    return pd.to_timedelta(pd.to_numeric(_series, errors='coerce'), unit=_unit_map.get(_unit, 'D'))",
            "def _k2p_add_offset(_series, _offset):",
            "    _dt = pd.to_datetime(_series, errors='coerce')",
            "    if isinstance(_offset, pd.Series):",
            "        return pd.Series([_d + _o if pd.notna(_d) and pd.notna(_o) else pd.NaT for _d, _o in zip(_dt, _offset)], index=_dt.index)",
            "    return _dt + _offset",
            "for _col in _shift_columns:",
            "    if _col not in out_df.columns:",
            "        continue",
            "    if str(_shift_mode).upper() == 'SHIFT_PERIOD_COLUMN' and _period_column in out_df.columns:",
            "        _offset = out_df[_period_column].map(_k2p_parse_period_offset)",
            "    elif str(_shift_mode).upper() == 'SHIFT_NUMERICAL_COLUMN' and _numerical_column in out_df.columns:",
            "        _offset = _k2p_numeric_date_delta(out_df[_numerical_column])",
            "    else:",
            "        _offset = _k2p_parse_period_offset(_period_value)",
            "    _target_col = _col if str(_replace_or_append).upper() == 'REPLACE' else f'{_col}{_suffix}'",
            "    out_df[_target_col] = _k2p_add_offset(out_df[_col], _offset)",
        ]


def parse_date_shifter_settings(node_dir: Optional[Path]) -> DateShifterSettings:
    return DateShifter.parse(node_dir)


def generate_imports() -> List[str]:
    return ["import pandas as pd", "import re"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_date_shifter_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']  # input table",
    ]
    lines.extend(DateShifter.emit(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "Date Shifter"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
