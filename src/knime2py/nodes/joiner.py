#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, split_out_imports


FACTORY = "org.knime.base.node.preproc.joiner3.Joiner3NodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.joiner3.Joiner3NodeFactory"
)


@dataclass
class JoinCriterion:
    left_column: Optional[str] = None
    right_column: Optional[str] = None
    left_row_id: bool = False
    right_row_id: bool = False


@dataclass
class ColumnSelection:
    selected: List[str] = field(default_factory=list)
    deselected: List[str] = field(default_factory=list)
    include_unknown: bool = True


@dataclass
class JoinerSettings:
    criteria: List[JoinCriterion] = field(default_factory=list)
    include_matches: bool = True
    include_left_unmatched: bool = False
    include_right_unmatched: bool = False
    suffix: str = " (Right)"
    left_columns: ColumnSelection = field(default_factory=ColumnSelection)
    right_columns: ColumnSelection = field(default_factory=ColumnSelection)


def _bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _indexed_entries(parent: Optional[ET._Element]) -> List[str]:
    if parent is None:
        return []
    vals: list[tuple[int, str]] = []
    for entry in parent.xpath("./*[local-name()='entry']"):
        key = entry.get("key")
        if key and key.isdigit():
            vals.append((int(key), str(entry.get("value", ""))))
    return [value for _, value in sorted(vals)]


def _parse_column_choice(parent: Optional[ET._Element], side: str) -> tuple[Optional[str], bool]:
    if parent is None:
        return None, False
    choice_el = first_el(parent, f"./*[local-name()='config' and @key='{side}TableColumnV2']")
    if choice_el is None:
        return None, False
    regular = first(choice_el, "./*[local-name()='entry' and @key='regularChoice']/@value")
    special = first(choice_el, "./*[local-name()='entry' and @key='specialChoice_Internals']/@value")
    return regular or None, (special or "").strip().upper() == "ROW_ID"


def _parse_selection(model_el: ET._Element, key: str) -> ColumnSelection:
    cfg = first_el(model_el, f"./*[local-name()='config' and @key='{key}']")
    if cfg is None:
        return ColumnSelection()
    selected_el = first_el(
        cfg,
        "./*[local-name()='config' and @key='manualFilter']"
        "/*[local-name()='config' and @key='manuallySelected']",
    )
    deselected_el = first_el(
        cfg,
        "./*[local-name()='config' and @key='manualFilter']"
        "/*[local-name()='config' and @key='manuallyDeselected']",
    )
    include_unknown = _bool(
        first(
            cfg,
            "./*[local-name()='config' and @key='manualFilter']"
            "/*[local-name()='entry' and @key='includeUnknownColumns']/@value",
        ),
        True,
    )
    return ColumnSelection(
        selected=_indexed_entries(selected_el),
        deselected=_indexed_entries(deselected_el),
        include_unknown=include_unknown,
    )


def parse_joiner_settings(node_dir: Optional[Path]) -> JoinerSettings:
    if not node_dir:
        return JoinerSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return JoinerSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return JoinerSettings()

    criteria: List[JoinCriterion] = []
    for crit_el in model_el.xpath(
        "./*[local-name()='config' and @key='matchingCriteria']/*[local-name()='config']"
    ):
        left_col, left_row_id = _parse_column_choice(crit_el, "left")
        right_col, right_row_id = _parse_column_choice(crit_el, "right")
        criteria.append(
            JoinCriterion(
                left_column=left_col,
                right_column=right_col,
                left_row_id=left_row_id,
                right_row_id=right_row_id,
            )
        )

    suffix_el = first_el(model_el, "./*[local-name()='entry' and @key='suffix']")
    suffix = suffix_el.get("value") if suffix_el is not None and suffix_el.get("value") is not None else " (Right)"

    return JoinerSettings(
        criteria=criteria,
        include_matches=_bool(first(model_el, "./*[local-name()='entry' and @key='includeMatchesInOutput']/@value"), True),
        include_left_unmatched=_bool(
            first(model_el, "./*[local-name()='entry' and @key='includeLeftUnmatchedInOutput']/@value"), False
        ),
        include_right_unmatched=_bool(
            first(model_el, "./*[local-name()='entry' and @key='includeRightUnmatchedInOutput']/@value"), False
        ),
        suffix=suffix,
        left_columns=_parse_selection(model_el, "leftColumnSelectionConfigV2"),
        right_columns=_parse_selection(model_el, "rightColumnSelectionConfigV2"),
    )


def generate_imports() -> List[str]:
    return ["import pandas as pd"]


def _join_how(cfg: JoinerSettings) -> str:
    if cfg.include_matches and cfg.include_left_unmatched and cfg.include_right_unmatched:
        return "outer"
    if cfg.include_matches and cfg.include_left_unmatched:
        return "left"
    if cfg.include_matches and cfg.include_right_unmatched:
        return "right"
    return "inner"


def _emit_join_code(cfg: JoinerSettings) -> List[str]:
    left_keys = [
        {"column": c.left_column, "row_id": c.left_row_id}
        for c in cfg.criteria
        if c.left_row_id or c.left_column
    ]
    right_keys = [
        {"column": c.right_column, "row_id": c.right_row_id}
        for c in cfg.criteria
        if c.right_row_id or c.right_column
    ]

    lines: List[str] = [
        f"_join_how = {repr(_join_how(cfg))}",
        f"_join_suffix = {repr(cfg.suffix)}",
        f"_left_selected = {repr(cfg.left_columns.selected)}",
        f"_left_deselected = {repr(cfg.left_columns.deselected)}",
        f"_left_include_unknown = {repr(cfg.left_columns.include_unknown)}",
        f"_right_selected = {repr(cfg.right_columns.selected)}",
        f"_right_deselected = {repr(cfg.right_columns.deselected)}",
        f"_right_include_unknown = {repr(cfg.right_columns.include_unknown)}",
        f"_left_keys = {repr(left_keys)}",
        f"_right_keys = {repr(right_keys)}",
        "",
        "def _jn_select_columns(df, selected, deselected, include_unknown):",
        "    existing = list(df.columns)",
        "    if selected:",
        "        cols = [c for c in selected if c in df.columns]",
        "        if include_unknown:",
        "            known = set(selected) | set(deselected)",
        "            cols.extend([c for c in existing if c not in known and c not in cols])",
        "    else:",
        "        cols = existing",
        "    if deselected:",
        "        cols = [c for c in cols if c not in set(deselected)]",
        "    return cols",
        "",
        "left_cols = _jn_select_columns(df_left, _left_selected, _left_deselected, _left_include_unknown)",
        "right_cols = _jn_select_columns(df_right, _right_selected, _right_deselected, _right_include_unknown)",
        "left_df = df_left[left_cols].copy()",
        "right_df = df_right[right_cols].copy()",
        "",
        "if not _left_keys or not _right_keys or len(_left_keys) != len(_right_keys):",
        "    out_df = left_df.copy()",
        "else:",
        "    _left_on = []",
        "    _right_on = []",
        "    for _i, (_lk, _rk) in enumerate(zip(_left_keys, _right_keys)):",
        "        _l_tmp = f'__k2p_join_left_{_i}__'",
        "        _r_tmp = f'__k2p_join_right_{_i}__'",
        "        left_df[_l_tmp] = df_left.index.astype('string') if _lk.get('row_id') else df_left[_lk['column']].astype('string')",
        "        right_df[_r_tmp] = df_right.index.astype('string') if _rk.get('row_id') else df_right[_rk['column']].astype('string')",
        "        _left_on.append(_l_tmp)",
        "        _right_on.append(_r_tmp)",
        "    _right_rename = {}",
        "    _left_visible = set(left_cols)",
        "    for _col in right_cols:",
        "        if _col in _left_visible:",
        "            _right_rename[_col] = f'{_col}{_join_suffix}'",
        "    if _right_rename:",
        "        right_df = right_df.rename(columns=_right_rename)",
        "    out_df = pd.merge(left_df, right_df, how=_join_how, left_on=_left_on, right_on=_right_on, sort=False)",
        "    out_df = out_df.drop(columns=_left_on + _right_on, errors='ignore')",
    ]
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_joiner_settings(ndir)

    left_src, left_port = in_ports[0] if in_ports else ("UNKNOWN", "1")
    right_src, right_port = in_ports[1] if len(in_ports) > 1 else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df_left = context['{left_src}:{left_port}']",
        f"df_right = context['{right_src}:{right_port}']",
    ]
    lines.extend(_emit_join_code(cfg))

    for p in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{p}'] = out_df")
    return lines


def get_name() -> str:
    return "Joiner"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)

    left_pair: Optional[Tuple[str, str]] = None
    right_pair: Optional[Tuple[str, str]] = None
    for src_id, e in (incoming or []):
        src_port = str(getattr(e, "source_port", "") or "1")
        tgt_port = str(getattr(e, "target_port", "") or "")
        if tgt_port == "1":
            left_pair = (str(src_id), src_port)
        elif tgt_port == "2":
            right_pair = (str(src_id), src_port)

    norm_in: List[Tuple[str, str]] = []
    if left_pair:
        norm_in.append(left_pair)
    if right_pair:
        norm_in.append(right_pair)
    if len(norm_in) < 2:
        norm_in = [(str(src), str(getattr(e, "source_port", "") or "1")) for src, e in (incoming or [])]

    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, norm_in, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
