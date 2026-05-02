#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, normalize_in_ports, split_out_imports


FACTORY = "org.knime.base.node.preproc.constantvalue.ConstantValueColumnNodeFactory2"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.constantvalue.ConstantValueColumnNodeFactory2"
)


@dataclass
class ConstantColumnSetting:
    mode: str = "APPEND"
    append_name: str = "New column"
    replace_name: Optional[str] = None
    cell_class: str = "org.knime.core.data.def.StringCell"
    is_null: bool = False
    custom_or_missing: str = "CUSTOM"
    value: Optional[str] = None


def _bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def parse_constant_value_column_settings(node_dir: Optional[Path]) -> List[ConstantColumnSetting]:
    if not node_dir:
        return [ConstantColumnSetting()]
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return [ConstantColumnSetting()]

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return [ConstantColumnSetting()]

    settings: List[ConstantColumnSetting] = []
    for cfg in model_el.xpath(
        "./*[local-name()='config' and @key='newColumnSettings']/*[local-name()='config']"
    ):
        settings.append(
            ConstantColumnSetting(
                mode=(first(cfg, "./*[local-name()='entry' and @key='replaceOrAppend']/@value") or "APPEND").upper(),
                append_name=first(cfg, "./*[local-name()='entry' and @key='columnNameToAppend']/@value") or "New column",
                replace_name=first(cfg, "./*[local-name()='entry' and @key='columnNameToReplace']/@value") or None,
                cell_class=first(
                    cfg,
                    "./*[local-name()='config' and @key='type']"
                    "/*[local-name()='entry' and @key='cell_class']/@value",
                )
                or "org.knime.core.data.def.StringCell",
                is_null=_bool(
                    first(
                        cfg,
                        "./*[local-name()='config' and @key='type']"
                        "/*[local-name()='entry' and @key='is_null']/@value",
                    ),
                    False,
                ),
                custom_or_missing=(
                    first(cfg, "./*[local-name()='entry' and @key='customOrMissingValue']/@value") or "CUSTOM"
                ).upper(),
                value=first(
                    cfg,
                    "./*[local-name()='config' and @key='customValueParameters']"
                    "/*[local-name()='entry' and @key='value']/@value",
                ),
            )
        )
    return settings or [ConstantColumnSetting()]


def generate_imports() -> List[str]:
    return ["import pandas as pd"]


def _python_literal_value(setting: ConstantColumnSetting) -> str:
    if setting.is_null or setting.custom_or_missing == "MISSING":
        return "pd.NA"

    raw = setting.value
    cell_class = setting.cell_class.rsplit(".", 1)[-1]
    if raw is None:
        return "pd.NA"
    if cell_class in {"IntCell", "LongCell"}:
        try:
            return repr(int(raw))
        except ValueError:
            return "pd.NA"
    if cell_class in {"DoubleCell", "FloatCell"}:
        try:
            return repr(float(raw))
        except ValueError:
            return "pd.NA"
    if cell_class == "BooleanCell":
        return "True" if str(raw).strip().lower() in {"1", "true", "yes", "y"} else "False"
    return repr(raw)


def _emit_constant_column_code(settings: List[ConstantColumnSetting]) -> List[str]:
    lines: List[str] = ["out_df = df.copy()"]
    for i, setting in enumerate(settings):
        target_name = setting.replace_name if setting.mode == "REPLACE" and setting.replace_name else setting.append_name
        lines.append(f"_constant_col_name_{i} = {repr(target_name)}")
        lines.append(f"_constant_value_{i} = {_python_literal_value(setting)}")
        lines.append(f"out_df[_constant_col_name_{i}] = _constant_value_{i}")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_constant_value_column_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']",
    ]
    lines.extend(_emit_constant_column_code(settings))

    for p in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{p}'] = out_df")
    return lines


def get_name() -> str:
    return "Constant Value Column Appender"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
