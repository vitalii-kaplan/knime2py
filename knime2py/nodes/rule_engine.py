#!/usr/bin/env python3
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (
    normalize_in_ports,
    iter_entries,
    all_values,
    collect_module_imports,
    split_out_imports,
)

# KNIME factory
RULE_ENGINE_FACTORY = "org.knime.base.node.rules.engine.RuleEngineNodeFactory"

def can_handle(node_type: Optional[str]) -> bool:
    return bool(node_type and node_type.endswith(RULE_ENGINE_FACTORY))


# ---------------------------------------------------------------------
# settings.xml → RuleEngineSettings
# ---------------------------------------------------------------------

@dataclass
class Rule:
    """A single parsed rule in the simple subset we support."""
    kind: str                 # "compare" | "like" | "true"
    col: Optional[str]        # column name (None for TRUE)
    op: Optional[str]         # one of >, >=, <, <=, =, ==, != (compare only)
    value: Optional[str]      # literal (string or number) or LIKE pattern
    outcome: str              # literal outcome (string)

@dataclass
class RuleEngineSettings:
    rules: List[Rule]
    append: bool
    new_col: Optional[str]
    replace_col: Optional[str]

# Helpers ----------------------------------------------------------------

_RULE_COMMENT = re.compile(r"^\s*//")
_RE_TRUE = re.compile(r'^\s*TRUE\s*=>\s*"(?P<out>.*)"\s*$', re.I)

# $col$ OP value => "out"
_RE_COMPARE = re.compile(
    r'^\s*\$(?P<col>[^$]+)\$\s*'
    r'(?P<op>>=|<=|==|=|!=|>|<)\s*'
    r'(?P<val>".*?"|\S+)\s*=>\s*"(?P<out>.*)"\s*$',
    re.I,
)

# $col$ LIKE "pattern" => "out"   (pattern may contain * wildcards)
_RE_LIKE = re.compile(
    r'^\s*\$(?P<col>[^$]+)\$\s+LIKE\s+"(?P<pat>.*)"\s*=>\s*"(?P<out>.*)"\s*$',
    re.I,
)

def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s

def _parse_one_rule(line: str) -> Optional[Rule]:
    """Parse one KNIME rule line (simple subset). Returns None for comments/blank."""
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
        val_raw = m.group("val").strip()
        out = _strip_quotes(m.group("out"))
        # store value normalized but keep original type guessing for codegen
        val = _strip_quotes(val_raw) if (val_raw.startswith(("'", '"')) and val_raw.endswith(("'", '"'))) else val_raw
        return Rule(kind="compare", col=col, op=op, value=val, outcome=out)

    m = _RE_LIKE.match(s)
    if m:
        col = m.group("col").strip()
        pat = m.group("pat")
        out = _strip_quotes(m.group("out"))
        return Rule(kind="like", col=col, op=None, value=pat, outcome=out)

    # Unknown/advanced constructs → skip for now (MVP)
    return None


def parse_rule_engine_settings(node_dir: Optional[Path]) -> RuleEngineSettings:
    if not node_dir:
        return RuleEngineSettings(rules=[], append=True, new_col=None, replace_col=None)

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return RuleEngineSettings(rules=[], append=True, new_col=None, replace_col=None)

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    # Get rules under model/rules as ordered <entry key="0" value="..."/>
    rules_raw: List[str] = []
    # XPath to 'rules' config
    rules_cfgs = root.xpath(
        ".//*[local-name()='config' and @key='model']"
        "/*[local-name()='config' and @key='rules']"
    )
    if rules_cfgs:
        # iterate entries by numeric key ordering if present
        entries = []
        for k, v in iter_entries(rules_cfgs[0]):
            if k.isdigit() and v is not None:
                entries.append((int(k), v))
        for _, v in sorted(entries, key=lambda t: t[0]):
            rules_raw.append(v)

    rules: List[Rule] = []
    for line in rules_raw:
        r = _parse_one_rule(line)
        if r:
            rules.append(r)

    # Target column behaviour
    model_cfgs = root.xpath(".//*[local-name()='config' and @key='model']")
    append = True
    new_col = None
    replace_col = None
    if model_cfgs:
        model = model_cfgs[0]
        # append-column true/false
        for k, v in iter_entries(model):
            lk = k.lower()
            if lk == "append-column":
                append = (str(v).strip().lower() == "true")
        # names
        new_col_vals = all_values(model, ".//*[local-name()='entry' and @key='new-column-name']/@value")
        if new_col_vals:
            new_col = (new_col_vals[0] or "").strip() or None
        rep_col_vals = all_values(model, ".//*[local-name()='entry' and @key='replace-column-name']/@value")
        if rep_col_vals:
            replace_col = (rep_col_vals[0] or "").strip() or None

    return RuleEngineSettings(rules=rules, append=append, new_col=new_col, replace_col=replace_col)


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    # Only pandas needed for emitted code
    return ["import pandas as pd"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.rules.engine.RuleEngineNodeFactory"
)

def _literal_py(val: str) -> str:
    """
    Return a Python literal for a KNIME rule value:
    - numeric strings → keep as-is (int/float) if they parse
    - otherwise → single-quoted Python string
    """
    s = str(val).strip()
    # Try int
    if re.fullmatch(r"[+-]?\d+", s):
        return s
    # Try float
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", s):
        return s
    # String literal
    return repr(s)

def _wildcard_to_regex(pat: str) -> str:
    # Convert simple * wildcards to a regex; escape other chars
    esc = re.escape(pat)
    return "^" + esc.replace(r"\*", ".*") + "$"

def _emit_rule_code(settings: RuleEngineSettings) -> List[str]:
    """
    Emit lines that:
      - build conditions cond0..condN in order
      - apply outcomes into res using .mask()
      - set default via .fillna() from TRUE if present
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    lines.append("res = pd.Series(pd.NA, index=out_df.index, dtype='object')")

    cond_idx = 0
    default_outcome: Optional[str] = None

    for r in settings.rules:
        if r.kind == "true":
            default_outcome = r.outcome
            continue

        if r.kind == "compare" and r.col and r.op and r.value is not None:
            col = r.col
            op = r.op
            val_lit = _literal_py(r.value)
            cond = f"cond{cond_idx}"
            # Normalize '=' to '==' for Python
            pyop = "==" if op == "=" else op
            lines.append(f"{cond} = (out_df[{repr(col)}] {pyop} {val_lit})")
            lines.append(f"res = res.mask({cond}, {repr(r.outcome)})")
            cond_idx += 1
            continue

        if r.kind == "like" and r.col and r.value is not None:
            col = r.col
            regex = _wildcard_to_regex(r.value)
            cond = f"cond{cond_idx}"
            # Use case-sensitive contains by default (KNIME has options; MVP keeps simple)
            lines.append(f"{cond} = out_df[{repr(col)}].astype('string').str.contains({repr(regex)}, regex=True, na=False)")
            lines.append(f"res = res.mask({cond}, {repr(r.outcome)})")
            cond_idx += 1
            continue

        # Unsupported rule → leave a breadcrumb
        lines.append(f"# TODO: unsupported rule skipped: {r}")

    if default_outcome is not None:
        lines.append(f"res = res.fillna({repr(default_outcome)})")

    # Choose target column
    target = settings.new_col if settings.append else settings.replace_col
    target = target or "RuleResult"
    lines.append(f"out_df[{repr(target)}] = res")

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

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_rule_code(cfg))

    # Publish to context
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
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Entry used by emitters:
      - returns (imports, body_lines) if this module can handle the node type
      - returns None otherwise
    """
    if not (ntype and can_handle(ntype)):
        return None

    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
