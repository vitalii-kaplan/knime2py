#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import (
    Rule,
    collect_module_imports,
    first,
    first_el,
    normalize_in_ports,
    rule_literal_py,
    rule_wildcard_to_regex,
    split_out_imports,
)
from .rule_engine import parse_rule_engine_settings


FACTORY = "org.knime.base.node.rules.engine.RuleEngineFilterNodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.rules.engine.RuleEngineFilterNodeFactory"
)


@dataclass
class RuleEngineFilterSettings:
    rules: List[Rule]
    include: bool = True


def _bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def parse_rule_engine_filter_settings(node_dir: Optional[Path]) -> RuleEngineFilterSettings:
    base = parse_rule_engine_settings(node_dir)
    include = True

    if node_dir:
        settings_path = node_dir / "settings.xml"
        if settings_path.exists():
            root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
            model = first_el(root, ".//*[local-name()='config' and @key='model']")
            if model is not None:
                include = _bool(first(model, "./*[local-name()='entry' and @key='include']/@value"), True)

    return RuleEngineFilterSettings(rules=base.rules, include=include)


def generate_imports() -> List[str]:
    return ["import pandas as pd"]


def _outcome_bool(outcome: str) -> Optional[bool]:
    val = str(outcome).strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n"}:
        return False
    return None


def _emit_rule_filter_code(settings: RuleEngineFilterSettings) -> List[str]:
    lines: List[str] = [
        "out_df = df.copy()",
        "keep = pd.Series(pd.NA, index=out_df.index, dtype='object')",
    ]

    idx = 0
    default_keep: Optional[bool] = None

    for rule in settings.rules:
        if rule.kind == "true":
            default_keep = _outcome_bool(rule.outcome)
            continue

        outcome = _outcome_bool(rule.outcome)
        if outcome is None:
            lines.append(f"# TODO: unsupported non-boolean filter outcome skipped: {rule.outcome!r}")
            continue

        if rule.kind == "compare" and rule.col and rule.op and rule.value is not None:
            cond = f"cond{idx}"
            pyop = "==" if rule.op == "=" else rule.op
            lines.append(f"{cond} = (out_df[{repr(rule.col)}] {pyop} {rule_literal_py(rule.value)})")
            lines.append(f"{cond} = {cond}.fillna(False)")
            lines.append(f"keep = keep.mask(keep.isna() & {cond}, {repr(outcome)})")
            idx += 1
            continue

        if rule.kind == "like" and rule.col and rule.value is not None:
            cond = f"cond{idx}"
            regex = rule_wildcard_to_regex(rule.value)
            lines.append(
                f"{cond} = out_df[{repr(rule.col)}].astype('string').str.contains({repr(regex)}, regex=True, na=False)"
            )
            lines.append(f"keep = keep.mask(keep.isna() & {cond}, {repr(outcome)})")
            idx += 1
            continue

        lines.append(f"# TODO: unsupported rule skipped: {rule}")

    if default_keep is None:
        default_keep = False
    lines.append(f"keep = keep.where(keep.notna(), {repr(default_keep)})")
    if not settings.include:
        lines.append("keep = ~keep.astype(bool)")
    lines.append("out_df = out_df[keep.astype(bool)].copy()")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_rule_engine_filter_settings(ndir)
    src_id, in_port = normalize_in_ports(in_ports)[0]

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{in_port}']  # input table",
    ]
    lines.extend(_emit_rule_filter_code(settings))

    for p in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{p}'] = out_df")
    return lines


def get_name() -> str:
    return "Rule-based Row Filter"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
