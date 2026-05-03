#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, normalize_in_ports, split_out_imports


FACTORY = "org.knime.base.node.preproc.sorter.SorterNodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.sorter.SorterNodeFactory"
)


@dataclass
class SortCriterion:
    column: str
    ascending: bool = True


@dataclass
class SorterSettings:
    criteria: List[SortCriterion] = field(default_factory=list)
    missing_to_end: bool = False


def _bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


class Sorter:
    @staticmethod
    def parse(node_dir: Optional[Path]) -> SorterSettings:
        if not node_dir:
            return SorterSettings()
        settings_path = node_dir / "settings.xml"
        if not settings_path.exists():
            return SorterSettings()

        root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
        model = first_el(root, ".//*[local-name()='config' and @key='model']")
        if model is None:
            return SorterSettings()

        criteria: List[SortCriterion] = []
        sorting_cfg = first_el(model, "./*[local-name()='config' and @key='sortingCriteria']")
        if sorting_cfg is not None:
            for cfg in sorting_cfg.xpath("./*[local-name()='config']"):
                column = first(
                    cfg,
                    "./*[local-name()='config' and @key='columnV2']"
                    "/*[local-name()='entry' and @key='regularChoice']/@value",
                )
                if not column:
                    continue
                order = first(cfg, "./*[local-name()='entry' and @key='sortingOrder']/@value") or "ASCENDING"
                criteria.append(SortCriterion(column=column, ascending=order.strip().upper() != "DESCENDING"))

        return SorterSettings(
            criteria=criteria,
            missing_to_end=_bool(first(model, "./*[local-name()='entry' and @key='missingToEnd']/@value"), False),
        )

    @staticmethod
    def emit(settings: SorterSettings) -> List[str]:
        return [
            f"_sort_columns = {repr([criterion.column for criterion in settings.criteria])}",
            f"_sort_ascending = {repr([criterion.ascending for criterion in settings.criteria])}",
            f"_missing_to_end = {repr(settings.missing_to_end)}",
            "_existing_sort = [(col, asc) for col, asc in zip(_sort_columns, _sort_ascending) if col in df.columns]",
            "if _existing_sort:",
            "    if len(_existing_sort) == 1 and not _existing_sort[0][1] and pd.api.types.is_timedelta64_dtype(df[_existing_sort[0][0]]):",
            "        _sort_col = _existing_sort[0][0]",
            "        _minutes = df[_sort_col].dt.total_seconds() / 60",
            "        _sign_rank = pd.Series(3, index=df.index)",
            "        _sign_rank[_minutes < 0] = 0",
            "        _sign_rank[_minutes > 0] = 1",
            "        _sign_rank[_minutes == 0] = 2",
            "        def _k2p_duration_sort_text(_value):",
            "            if pd.isna(_value):",
            "                return ''",
            "            _total_minutes = int(pd.Timedelta(_value).total_seconds() // 60)",
            "            if _total_minutes == 0:",
            "                return 'PT0S'",
            "            _sign = -1 if _total_minutes < 0 else 1",
            "            _hours, _mins = divmod(abs(_total_minutes), 60)",
            "            if _sign < 0:",
            "                if _hours and _mins:",
            "                    return f'PT-{_hours}H-{_mins}M'",
            "                if _hours:",
            "                    return f'PT-{_hours}H'",
            "                return f'PT-{_mins}M'",
            "            if _hours and _mins:",
            "                return f'PT{_hours}H{_mins}M'",
            "            if _hours:",
            "                return f'PT{_hours}H'",
            "            return f'PT{_mins}M'",
            "        def _k2p_natural_key(_text):",
            "            return re.sub(r'\\d+', lambda _m: f'{int(_m.group()):012d}', str(_text))",
            "        _duration_text = df[_sort_col].map(_k2p_duration_sort_text).map(_k2p_natural_key)",
            "        out_df = (",
            "            df.assign(_k2p_sign_rank=_sign_rank, _k2p_duration_text=_duration_text)",
            "            .sort_values(['_k2p_sign_rank', '_k2p_duration_text'], ascending=[True, False], kind='mergesort')",
            "            .drop(columns=['_k2p_sign_rank', '_k2p_duration_text'])",
            "            .reset_index(drop=True)",
            "        )",
            "    else:",
            "        out_df = df.sort_values(",
            "            by=[col for col, _asc in _existing_sort],",
            "            ascending=[asc for _col, asc in _existing_sort],",
            "            na_position='last' if _missing_to_end else 'first',",
            "            kind='mergesort',",
            "        ).reset_index(drop=True)",
            "else:",
            "    out_df = df.copy()",
        ]


def parse_sorter_settings(node_dir: Optional[Path]) -> SorterSettings:
    return Sorter.parse(node_dir)


def generate_imports() -> List[str]:
    return ["import pandas as pd", "import re"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_sorter_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']  # input table",
    ]
    lines.extend(Sorter.emit(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "Sorter"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
