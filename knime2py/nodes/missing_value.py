#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from . import node_utils as U  # reuse regex-friendly XML helpers (_iter_entries etc.)

# KNIME factory for this node
MISSING_VALUE_FACTORY = (
    "org.knime.base.node.preproc.pmml.missingval.compute.MissingValueHandlerNodeFactory"
)

def can_handle(node_type: Optional[str]) -> bool:
    return bool(node_type and node_type.endswith(MISSING_VALUE_FACTORY))


# ---------------------------------------------------------------------
# settings.xml → MissingValueSettings
# ---------------------------------------------------------------------

@dataclass
class TypePolicy:
    dtype: str                 # "int", "float", "string", "boolean", etc. (normalized)
    strategy: str              # "fixed" | "mean" | "median" | "mode" | "ffill" | "bfill" | "drop"
    value: Optional[str] = None  # fixed value as string (we'll cast later in code)

@dataclass
class MissingValueSettings:
    by_dtype: List[TypePolicy] = field(default_factory=list)
    # (Future: add per-column policies if needed)


# Map KNIME Java cell classes → simple dtype token
_CELL_TO_DTYPE = {
    "org.knime.core.data.def.IntCell": "int",
    "org.knime.core.data.def.LongCell": "int",
    "org.knime.core.data.def.DoubleCell": "float",
    "org.knime.core.data.def.StringCell": "string",
    "org.knime.core.data.def.BooleanCell": "boolean",
}

# Heuristics: factoryID substrings → strategy token
def _strategy_from_factory(factory_id: str) -> str:
    s = factory_id.lower()
    if "fixed" in s:
        return "fixed"
    if "mean" in s:
        return "mean"
    if "median" in s:
        return "median"
    if "mode" in s or "mostfreq" in s:
        return "mode"
    if "previous" in s or "prev" in s or "forward" in s or "ffill" in s:
        return "ffill"
    if "next" in s or "backward" in s or "bfill" in s:
        return "bfill"
    if "remove" in s and "row" in s:
        return "drop"
    # Fallback
    return "fixed"

# KNIME uses different keys for the fixed values; try a few obvious ones
_FIXED_VALUE_KEYS = (
    "fixIntegerValue", "fixLongValue", "fixDoubleValue",
    "fixStringValue", "fixBooleanValue", "fixValue"
)

def _first_present_value(cfg: ET._Element, keys=_FIXED_VALUE_KEYS) -> Optional[str]:
    for k, v in U._iter_entries(cfg):
        if k in keys:
            return v
    return None


def parse_missing_value_settings(node_dir: Optional[Path]) -> MissingValueSettings:
    """
    Parse the Missing Value node settings, focusing on type-wide policies under:
      model/dataTypeSettings/<config key='org.knime.core.data.def.*Cell'>
        - entry key="factoryID" value="..."
        - config key="settings"... (may contain a fixed value entry)
    """
    if not node_dir:
        return MissingValueSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return MissingValueSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    # Find dataTypeSettings container (if present)
    dts = root.xpath(
        ".//*[local-name()='config' and @key='model']"
        "/*[local-name()='config' and @key='dataTypeSettings']"
    )
    if not dts:
        return MissingValueSettings()

    by_dtype: List[TypePolicy] = []

    for cfg in dts[0].xpath("./*[local-name()='config']"):
        # e.g. key="org.knime.core.data.def.IntCell"
        cell_cls = (cfg.get("key") or "").strip()
        dtype = _CELL_TO_DTYPE.get(cell_cls)
        if not dtype:
            continue

        # factoryID decides the strategy
        factory_id = None
        for k, v in U._iter_entries(cfg):
            if k == "factoryID":
                factory_id = v
                break
        if not factory_id:
            continue

        strategy = _strategy_from_factory(factory_id)

        # optional fixed value (for "fixed" strategy)
        fixed_val = None
        for sub in cfg.xpath("./*[local-name()='config' and @key='settings']"):
            fixed_val = _first_present_value(sub)
            if fixed_val is not None:
                break

        by_dtype.append(TypePolicy(dtype=dtype, strategy=strategy, value=fixed_val))

    return MissingValueSettings(by_dtype=by_dtype)


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.pmml.missingval.compute.MissingValueHandlerNodeFactory"
)

def _emit_fill_code(settings: MissingValueSettings) -> List[str]:
    """
    Produce python lines that transform `df` into `out_df` according to dtype policies.
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")

    # Build groups per strategy to keep code tight
    ints: List[TypePolicy] = [p for p in settings.by_dtype if p.dtype == "int"]
    floats: List[TypePolicy] = [p for p in settings.by_dtype if p.dtype == "float"]
    strings: List[TypePolicy] = [p for p in settings.by_dtype if p.dtype == "string"]
    booleans: List[TypePolicy] = [p for p in settings.by_dtype if p.dtype == "boolean"]

    def _emit_for_group(group: List[TypePolicy], sel: str, caster: Optional[str] = None):
        for pol in group:
            strat = pol.strategy
            if strat == "fixed":
                raw = pol.value
                # Prefer literal emission when possible
                if caster == "int" and raw is not None:
                    s = str(raw).strip()
                    # handle +/- integers as literals
                    if s.lstrip("+-").isdigit():
                        lines.append(f"out_df[{sel}] = out_df[{sel}].fillna({int(s)})")
                    else:
                        lines.append(f"out_df[{sel}] = out_df[{sel}].fillna(int({repr(raw)}))")
                elif caster is None and raw is not None:
                    # strings/booleans/floats – emit raw literal if it parses cleanly for floats
                    s = str(raw).strip()
                    try:
                        fval = float(s)
                        # If it's an integer-like float (e.g., "0"), keep as 0.0 to be safe for float columns
                        lines.append(f"out_df[{sel}] = out_df[{sel}].fillna({fval})")
                    except ValueError:
                        lines.append(f"out_df[{sel}] = out_df[{sel}].fillna({repr(raw)})")
                else:
                    lines.append(f"out_df[{sel}] = out_df[{sel}].fillna(None)")
            elif strat == "mean":
                lines.append(f"out_df[{sel}] = out_df[{sel}].apply(lambda s: s.fillna(s.mean()))")
            elif strat == "median":
                lines.append(f"out_df[{sel}] = out_df[{sel}].apply(lambda s: s.fillna(s.median()))")
            elif strat == "mode":
                # use first mode if available
                lines.append(f"out_df[{sel}] = out_df[{sel}].apply(lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else s))")
            elif strat == "ffill":
                lines.append(f"out_df[{sel}] = out_df[{sel}].ffill()")
            elif strat == "bfill":
                lines.append(f"out_df[{sel}] = out_df[{sel}].bfill()")
            elif strat == "drop":
                lines.append(f"out_df = out_df.dropna(subset=out_df[{sel}].columns.tolist())")
            else:
                lines.append(f"# TODO: unsupported strategy '{strat}' for {sel}; leaving as-is")

    # Integer-like columns (nullable Int dtypes + classic ints)
    lines.append("int_cols = out_df.select_dtypes(include=['Int64', 'Int32', 'Int16', 'int64', 'int32', 'int16']).columns")
    _emit_for_group(ints, "int_cols", caster="int")

    # Float columns
    lines.append("float_cols = out_df.select_dtypes(include=['float64', 'float32']).columns")
    _emit_for_group(floats, "float_cols", caster=None)

    # String/object columns (treat pandas 'string' & 'object')
    lines.append("str_cols = out_df.select_dtypes(include=['string', 'object']).columns")
    _emit_for_group(strings, "str_cols", caster=None)

    # Boolean columns
    lines.append("bool_cols = out_df.select_dtypes(include=['boolean', 'bool']).columns")
    _emit_for_group(booleans, "bool_cols", caster=None)

    if not settings.by_dtype:
        lines.append("# No missing-value policies found; passthrough.")
        lines.append("out_df = df")

    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Emit body lines for the .py workbook.
    """
    ndir = Path(node_dir) if node_dir else None
    settings = parse_missing_value_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = U.normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_fill_code(settings))

    # Publish to context (default port '1')
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
