#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, normalize_in_ports, split_out_imports


FACTORY = "org.knime.base.node.preproc.duplicates.DuplicateRowFilterNodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.duplicates.DuplicateRowFilterNodeFactory"
)


@dataclass
class DuplicateRowFilterSettings:
    group_columns: List[str] = field(default_factory=list)
    remove_duplicates: bool = True
    add_row_duplicate_flag: bool = False
    duplicate_flag_column_name: str = "Duplicate Status"
    add_row_id_flag: bool = False
    row_id_flag_column_name: str = "Duplicate Chosen"
    row_selection: str = "FIRST"
    reference_column: str = ""
    retain_order: bool = True


def _bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _included_names(config) -> List[str]:
    included = first_el(config, "./*[local-name()='config' and @key='included_names']")
    if included is None:
        return []
    entries = included.xpath("./*[local-name()='entry' and @key!='array-size']/@value")
    return [str(value) for value in entries if str(value)]


class DuplicateRowFilter:
    @staticmethod
    def parse(node_dir: Optional[Path]) -> DuplicateRowFilterSettings:
        if not node_dir:
            return DuplicateRowFilterSettings()
        settings_path = node_dir / "settings.xml"
        if not settings_path.exists():
            return DuplicateRowFilterSettings()

        root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
        model = first_el(root, ".//*[local-name()='config' and @key='model']")
        if model is None:
            return DuplicateRowFilterSettings()

        group_cfg = first_el(model, "./*[local-name()='config' and @key='group_cols']")
        group_columns = _included_names(group_cfg) if group_cfg is not None else []

        return DuplicateRowFilterSettings(
            group_columns=group_columns,
            remove_duplicates=_bool(first(model, "./*[local-name()='entry' and @key='remove_duplicates']/@value"), True),
            add_row_duplicate_flag=_bool(
                first(model, "./*[local-name()='entry' and @key='add_row_duplicate_flag']/@value"), False
            ),
            duplicate_flag_column_name=first(
                model, "./*[local-name()='entry' and @key='unique_flag_column_name']/@value"
            )
            or "Duplicate Status",
            add_row_id_flag=_bool(first(model, "./*[local-name()='entry' and @key='add_row_id_flag']/@value"), False),
            row_id_flag_column_name=first(
                model, "./*[local-name()='entry' and @key='row_id_flag_column_name']/@value"
            )
            or "Duplicate Chosen",
            row_selection=first(model, "./*[local-name()='entry' and @key='row_selection']/@value") or "FIRST",
            reference_column=first(model, "./*[local-name()='entry' and @key='reference_col']/@value") or "",
            retain_order=_bool(first(model, "./*[local-name()='entry' and @key='retain_order']/@value"), True),
        )

    @staticmethod
    def emit(settings: DuplicateRowFilterSettings) -> List[str]:
        return [
            f"_group_columns = {repr(settings.group_columns)}",
            f"_remove_duplicates = {repr(settings.remove_duplicates)}",
            f"_add_duplicate_flag = {repr(settings.add_row_duplicate_flag)}",
            f"_duplicate_flag_col = {repr(settings.duplicate_flag_column_name)}",
            f"_add_row_id_flag = {repr(settings.add_row_id_flag)}",
            f"_row_id_flag_col = {repr(settings.row_id_flag_column_name)}",
            f"_row_selection = {repr(settings.row_selection)}",
            f"_reference_col = {repr(settings.reference_column)}",
            "_subset = [col for col in _group_columns if col in df.columns]",
            "out_df = df.copy()",
            "if _subset:",
            "    _selection = str(_row_selection).upper()",
            "    if _selection == 'LAST':",
            "        _keep = 'last'",
            "    elif _selection == 'NONE':",
            "        _keep = False",
            "    else:",
            "        _keep = 'first'",
            "    if _remove_duplicates:",
            "        out_df = out_df.drop_duplicates(subset=_subset, keep=_keep).reset_index(drop=True)",
            "    else:",
            "        _duplicated = out_df.duplicated(subset=_subset, keep=False)",
            "        if _add_duplicate_flag:",
            "            out_df[_duplicate_flag_col] = np.where(_duplicated, 'Duplicate', 'Unique')",
            "        if _add_row_id_flag:",
            "            _chosen = ~out_df.duplicated(subset=_subset, keep=_keep if _keep else False)",
            "            out_df[_row_id_flag_col] = np.where(_chosen, 'Chosen', '')",
            "else:",
            "    out_df = out_df.reset_index(drop=True)",
        ]


def parse_duplicate_row_filter_settings(node_dir: Optional[Path]) -> DuplicateRowFilterSettings:
    return DuplicateRowFilter.parse(node_dir)


def generate_imports() -> List[str]:
    return ["import numpy as np", "import pandas as pd"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_duplicate_row_filter_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']  # input table",
    ]
    lines.extend(DuplicateRowFilter.emit(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "Duplicate Row Filter"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
