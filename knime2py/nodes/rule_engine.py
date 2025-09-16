#!/usr/bin/env python3

####################################################################################################
#
# Rule Engine
#
# Applies a subset of KNIME Rule Engine logic to an input table and writes the result to context.
# Parses settings.xml rules and emits pandas code that evaluates them in order and assigns outcomes.
#
# - Supported rules: TRUE => "out"; $col$ <op> value => "out" with <, <=, >, >=, =, ==, !=;
#   $col$ LIKE "pat" (uses * as wildcard; converted to a regex). A trailing TRUE acts as default.
# - Column output: append to a new column if configured; otherwise replace the specified column;
#   falls back to "RuleResult" when no name is provided.
# - Literals: numeric strings are emitted as numbers; everything else is a quoted Python literal.
# - Limitations: no AND/OR chaining, no between/in lists, no regex beyond LIKE→wildcard, and no
#   type coercion beyond basic string/number handling.
#
####################################################################################################


from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *

# ---------------------------------------------------------------------
# Node identity
# ---------------------------------------------------------------------

FACTORY = "org.knime.base.node.rules.engine.RuleEngineNodeFactory"

# ---------------------------------------------------------------------
# settings.xml → RuleEngineSettings
# ---------------------------------------------------------------------

@dataclass
class Rule:
    kind: str                 # "compare" | "like" | "true"
    col: Optional[str]        # column name (None for TRUE)
    op: Optional[str]         # >, >=, <, <=, =, ==, != (compare only)
    value: Optional[str]      # literal value (or LIKE pattern)
    outcome: str              # literal outcome

@dataclass
class RuleEngineSettings:
    rules: List[Rule]
    append: bool
    new_col: Optional[str]
    replace_col: Optional[str]

# ---- Parsers for the simple rule subset ----

_RULE_COMMENT = re.compile(r"^\s*//")
_RE_TRUE = re.compile(r'^\s*TRUE\s*=>\s*"(?P<out>.*)"\s*$', re.I)
_RE_COMPARE = re.compile(
    r'^\s*\$(?P<col>[^$]+)\$\s*'
    r'(?P<op>>=|<=|==|=|!=|>|<)\s*'
    r'(?P<val>".*?"|\S+)\s*=>\s*"(?P<out>.*)"\s*$',
    re.I,
)
_RE_LIKE = re.compile(
    r'^\s*\$(?P<col>[^$]+)\$\s+LIKE\s+"(?P<pat>.*)"\s*=>\s*"(?P<out>.*)"\s*$',
    re.I,
)

def _strip_quotes(s: str) -> str:
    return s[1:-1] if (len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"") else s

def _parse_one_rule(line: str) -> Optional[Rule]:
    s = html.unescape(line or "").strip()
    if not s or _RULE_COMMENT.match(s):
        return None
    m = _RE_TRUE.match(s)
    if m:
        return Rule(kind="true", col=None, op=None, value=None, outcome=_strip_quotes(m.group("out")))
    m = _RE_COMPARE.match(s)
    if m:
        col = m.group("col").strip()
        op = m.group("op")
        vraw = m.group("val").strip()
        val = _strip_quotes(vraw) if (vraw[:1] in "'\"" and vraw[-1:] in "'\"") else vraw
        return Rule(kind="compare", col=col, op=op, value=val, outcome=_strip_quotes(m.group("out")))
    m = _RE_LIKE.match(s)
    if m:
        return Rule(kind="like", col=m.group("col").strip(), op=None,
                    value=m.group("pat"), outcome=_strip_quotes(m.group("out")))
    return None

def parse_rule_engine_settings(node_dir: Optional[Path]) -> RuleEngineSettings:
    if not node_dir:
        return RuleEngineSettings(rules=[], append=True, new_col=None, replace_col=None)

    sp = node_dir / "settings.xml"
    if not sp.exists():
        return RuleEngineSettings(rules=[], append=True, new_col=None, replace_col=None)

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()

    # model + rules blocks
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    rules_cfg = first_el(root, ".//*[local-name()='config' and @key='model']"
                              "/*[local-name()='config' and @key='rules']")

    rules_raw: List[str] = []
    if rules_cfg is not None:
        numbered: List[tuple[int, str]] = []
        for k, v in iter_entries(rules_cfg):
            if k.isdigit() and v is not None:
                numbered.append((int(k), v))
        for _, v in sorted(numbered, key=lambda t: t[0]):
            rules_raw.append(v)

    rules: List[Rule] = []
    for line in rules_raw:
        r = _parse_one_rule(line)
        if r:
            rules.append(r)

    # column handling (append vs replace) + names
    append = True
    new_col = replace_col = None
    if model is not None:
        av = all_values(model, ".//*[local-name()='entry' and @key='append-column']/@value")
        if av:
            append = (av[0].strip().lower() == "true")
        new_col = first(model, ".//*[local-name()='entry' and @key='new-column-name']/@value")
        replace_col = first(model, ".//*[local-name()='entry' and @key='replace-column-name']/@value")

    return RuleEngineSettings(
        rules=rules,
        append=append,
        new_col=(new_col or None),
        replace_col=(replace_col or None),
    )

# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.rules.engine.RuleEngineNodeFactory"
)

def _literal_py(val: str) -> str:
    s = str(val).strip()
    if re.fullmatch(r"[+-]?\d+", s):
        return s
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", s):
        return s
    return repr(s)

def _wildcard_to_regex(pat: str) -> str:
    esc = re.escape(pat)
    return "^" + esc.replace(r"\*", ".*") + "$"

def _emit_rule_code(settings: RuleEngineSettings) -> List[str]:
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    lines.append("res = pd.Series(pd.NA, index=out_df.index, dtype='object')")

    idx = 0
    default_outcome: Optional[str] = None

    for r in settings.rules:
        if r.kind == "true":
            default_outcome = r.outcome
            continue
        if r.kind == "compare" and r.col and r.op and (r.value is not None):
            cond = f"cond{idx}"
            pyop = "==" if r.op == "=" else r.op
            lines.append(f"{cond} = (out_df[{repr(r.col)}] {pyop} {_literal_py(r.value)})")
            lines.append(f"res = res.mask({cond}, {repr(r.outcome)})")
            idx += 1
            continue
        if r.kind == "like" and r.col and (r.value is not None):
            cond = f"cond{idx}"
            regex = _wildcard_to_regex(r.value)
            lines.append(f"{cond} = out_df[{repr(r.col)}].astype('string').str.contains({repr(regex)}, regex=True, na=False)")
            lines.append(f"res = res.mask({cond}, {repr(r.outcome)})")
            idx += 1
            continue
        lines.append(f"# TODO: unsupported rule skipped: {r}")

    if default_outcome is not None:
        lines.append(f"res = res.fillna({repr(default_outcome)})")

    target = settings.new_col if settings.append else settings.replace_col
    lines.append(f"out_df[{repr(target or 'RuleResult')}] = res")
    return lines

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_rule_engine_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    src_id, in_port = normalize_in_ports(in_ports)[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_rule_code(cfg))

    ports = out_ports or ["1"]
    for p in sorted({(p or '1') for p in ports}):
        lines.append(f"context['{node_id}:{p}'] = out_df")
    return lines

def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    return "\n".join(generate_py_body(node_id, node_dir, in_ports, out_ports)) + "\n"

def handle(ntype, nid, npath, incoming, outgoing):
    """
    Return (imports, body_lines) if we can handle this node; otherwise None.
    """

    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
