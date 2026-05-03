#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, normalize_in_ports, split_out_imports


FACTORY = "org.knime.base.node.preproc.cellsplit2.CellSplitter2NodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.cellsplit2.CellSplitter2NodeFactory"
)


@dataclass
class CellSplitterSettings:
    column: str = ""
    remove_input_column: bool = False
    delimiter: str = ","
    remove_whitespaces: bool = True
    output_as_columns: bool = True
    number_of_columns: int = 2


def _bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _int(raw: Optional[str], default: int) -> int:
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


class CellSplitter:
    @staticmethod
    def parse(node_dir: Optional[Path]) -> CellSplitterSettings:
        if not node_dir:
            return CellSplitterSettings()
        settings_path = node_dir / "settings.xml"
        if not settings_path.exists():
            return CellSplitterSettings()

        root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
        model = first_el(root, ".//*[local-name()='config' and @key='model']")
        if model is None:
            return CellSplitterSettings()

        return CellSplitterSettings(
            column=first(model, "./*[local-name()='entry' and @key='colName']/@value") or "",
            remove_input_column=_bool(
                first(model, "./*[local-name()='entry' and @key='removeInputColumn']/@value"), False
            ),
            delimiter=first(model, "./*[local-name()='entry' and @key='delimiter']/@value") or ",",
            remove_whitespaces=_bool(
                first(model, "./*[local-name()='entry' and @key='removeWhitespaces']/@value"), True
            ),
            output_as_columns=_bool(
                first(model, "./*[local-name()='entry' and @key='outputAsColumns']/@value"), True
            ),
            number_of_columns=_int(first(model, "./*[local-name()='entry' and @key='numberOfCols']/@value"), 2),
        )

    @staticmethod
    def emit(settings: CellSplitterSettings) -> List[str]:
        return [
            f"_split_col = {repr(settings.column)}",
            f"_delimiter = {repr(settings.delimiter)}",
            f"_remove_input_col = {repr(settings.remove_input_column)}",
            f"_remove_whitespaces = {repr(settings.remove_whitespaces)}",
            f"_output_as_columns = {repr(settings.output_as_columns)}",
            f"_number_of_columns = {repr(settings.number_of_columns)}",
            "out_df = df.copy()",
            "if _split_col in out_df.columns:",
            "    _source = out_df[_split_col]",
            "    if pd.api.types.is_datetime64_any_dtype(_source):",
            "        _source = _source.dt.strftime('%Y-%m-%dT%H:%M')",
            "    else:",
            "        _source = _source.astype('string')",
            "    _parts = _source.str.split(_delimiter, n=max(_number_of_columns - 1, 0), expand=True)",
            "    if _remove_whitespaces:",
            "        _parts = _parts.apply(lambda _series: _series.str.strip())",
            "    if _output_as_columns:",
            "        for _idx in range(_parts.shape[1]):",
            "            _part = _parts[_idx]",
            "            _numeric = pd.to_numeric(_part, errors='coerce')",
            "            if _part.notna().any() and _numeric[_part.notna()].notna().all() and ((_numeric.dropna() % 1) == 0).all():",
            "                out_df[f'{_split_col}_Arr[{_idx}]'] = _numeric.astype('Int64')",
            "            else:",
            "                out_df[f'{_split_col}_Arr[{_idx}]'] = _part",
            "        if _remove_input_col:",
            "            out_df = out_df.drop(columns=[_split_col])",
            "    else:",
            "        out_df[f'{_split_col}_Arr'] = _parts.apply(lambda _row: [_v for _v in _row.tolist() if pd.notna(_v)], axis=1)",
            "        if _remove_input_col:",
            "            out_df = out_df.drop(columns=[_split_col])",
        ]


def parse_cell_splitter_settings(node_dir: Optional[Path]) -> CellSplitterSettings:
    return CellSplitter.parse(node_dir)


def generate_imports() -> List[str]:
    return ["import pandas as pd"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_cell_splitter_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']  # input table",
    ]
    lines.extend(CellSplitter.emit(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "Cell Splitter"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
