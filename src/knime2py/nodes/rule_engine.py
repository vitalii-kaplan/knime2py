#!/usr/bin/env python3

"""
Rule Engine Module.

Overview
----------------------------
This module applies a subset of KNIME Rule Engine logic to an input table and emits
pandas code that evaluates rules defined in a settings.xml file.

Runtime Behavior
----------------------------
Inputs:
- Reads a DataFrame from the context using the specified input port.

Outputs:
- Writes the resulting DataFrame back to the context, mapping it to the specified output
  ports.

Key algorithms:
- Evaluates rules in order, assigning outcomes based on comparisons and pattern matching.

Edge Cases
----------------------------
- Handles empty or constant columns, NaNs, and provides fallback paths for unsupported rules.

Generated Code Dependencies
----------------------------
- The generated code requires pandas for DataFrame manipulation.

Usage
----------------------------
- Typically invoked by the knime2py emitter in a KNIME workflow.
- Example context access: `df = context['input_table:1']`.

Node Identity
----------------------------
- KNIME factory id: `FACTORY = "org.knime.base.node.rules.engine.RuleEngineNodeFactory"`.

Configuration
----------------------------
- Settings are defined in the `RuleEngineSettings` dataclass, which includes:
  - `rules`: List of rules to evaluate.
  - `append`: Whether to append results to a new column (default: True).
  - `new_col`: Name of the new column for results (default: None).
  - `replace_col`: Name of the column to replace with results (default: None).
- Values are extracted from the settings.xml file using XPath queries.

Limitations / Not implemented
----------------------------
- Does not support AND/OR chaining, between/in lists, or regex beyond LIKE wildcard.

References
----------------------------
- For more information, visit: https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
  org.knime.base.node.rules.engine.RuleEngineNodeFactory
"""

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
class RuleEngineSettings:
    rules: List[Rule]
    append: bool
    new_col: Optional[str]
    replace_col: Optional[str]

def parse_rule_engine_settings(node_dir: Optional[Path]) -> RuleEngineSettings:
    """Parse the settings.xml file and return RuleEngineSettings."""
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

    rules = parse_knime_rules_from_config(rules_cfg)

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
    """Generate a list of imports required for the generated code."""
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.rules.engine.RuleEngineNodeFactory"
)

def _emit_rule_code(settings: RuleEngineSettings) -> List[str]:
    """Generate the code that evaluates the rules defined in RuleEngineSettings."""
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
            lines.append(f"{cond} = (out_df[{repr(r.col)}] {pyop} {rule_literal_py(r.value)})")
            lines.append(f"res = res.mask({cond}, {rule_literal_py(r.outcome)})")
            idx += 1
            continue
        if r.kind == "like" and r.col and (r.value is not None):
            cond = f"cond{idx}"
            regex = rule_wildcard_to_regex(r.value)
            lines.append(f"{cond} = out_df[{repr(r.col)}].astype('string').str.contains({repr(regex)}, regex=True, na=False)")
            lines.append(f"res = res.mask({cond}, {rule_literal_py(r.outcome)})")
            idx += 1
            continue
        lines.append(f"# TODO: unsupported rule skipped: {r}")

    if default_outcome is not None:
        literal = rule_literal_py(default_outcome)
        lines.append(f"res = res.where(res.notna(), {literal})")
        lines.append("res = res.infer_objects()")

    target = settings.new_col if settings.append else settings.replace_col
    lines.append(f"out_df[{repr(target or 'RuleResult')}] = res")
    return lines

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """Generate the Python code body for the node based on its configuration and input ports."""
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



def get_name() -> str:
    """Return name of the node in KNIME workflow."""
    return "Rule Engine"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handle the node and return (imports, body_lines) if we can handle this node; otherwise None.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
