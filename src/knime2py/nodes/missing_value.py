#!/usr/bin/env python3

####################################################################################################
#
# Missing Value Handler
#
# Applies KNIME Missing Value type-wide policies to an input table and writes the result to the
# node’s context. Parses settings.xml to map Java cell types to simple dtypes and derives a fill
# strategy per dtype, then generates pandas code to impute/propagate/drop accordingly.
#
# - Integers: mean/median/mode fills are rounded and per-column recast to nullable Int64.
# - Skip branches always contain an executable statement (pass) to avoid IndentationError.
# - We never emit .fillna(None).
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # iter_entries, first_el, first, collect_module_imports, split_out_imports, normalize_in_ports

FACTORY = "org.knime.base.node.preproc.pmml.missingval.compute.MissingValueHandlerNodeFactory"

# ---------------------------------------------------------------------
# settings.xml → MissingValueSettings
# ---------------------------------------------------------------------

@dataclass
class TypePolicy:
    dtype: str                 # "int", "float", "string", "boolean"
    strategy: str              # "fixed" | "mean" | "median" | "mode" | "ffill" | "bfill" | "drop"
    value: Optional[str] = None  # fixed value as string

@dataclass
class MissingValueSettings:
    by_dtype: List[TypePolicy] = field(default_factory=list)

_CELL_TO_DTYPE = {
    "org.knime.core.data.def.IntCell": "int",
    "org.knime.core.data.def.LongCell": "int",
    "org.knime.core.data.def.DoubleCell": "float",
    "org.knime.core.data.def.StringCell": "string",
    "org.knime.core.data.def.BooleanCell": "boolean",
}

def _strategy_from_factory(factory_id: str) -> str:
    """Determine the fill strategy based on the factory ID."""
    s = factory_id.lower()
    if "fixed" in s: return "fixed"
    if "mean" in s:  return "mean"
    if "median" in s:return "median"
    if "mode" in s or "mostfreq" in s: return "mode"
    if any(k in s for k in ("previous","prev","forward","ffill")): return "ffill"
    if any(k in s for k in ("next","backward","bfill")): return "bfill"
    if "remove" in s and "row" in s: return "drop"
    return "fixed"

_FIXED_VALUE_KEYS = (
    "fixIntegerValue", "fixLongValue", "fixDoubleValue",
    "fixStringValue", "fixBooleanValue", "fixValue"
)

def _first_present_value(cfg: ET._Element, keys=_FIXED_VALUE_KEYS) -> Optional[str]:
    """Retrieve the first present value from the configuration based on specified keys."""
    for k, v in iter_entries(cfg):
        if k in keys:
            return v
    return None

def parse_missing_value_settings(node_dir: Optional[Path]) -> MissingValueSettings:
    """Parse the missing value settings from the settings.xml file."""
    if not node_dir:
        return MissingValueSettings()
    sp = node_dir / "settings.xml"
    if not sp.exists():
        return MissingValueSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    dts = root.xpath(
        ".//*[local-name()='config' and @key='model']"
        "/*[local-name()='config' and @key='dataTypeSettings']"
    )
    if not dts:
        return MissingValueSettings()

    by_dtype: List[TypePolicy] = []
    for cfg in dts[0].xpath("./*[local-name()='config']"):
        cell_cls = (cfg.get("key") or "").strip()
        dtype = _CELL_TO_DTYPE.get(cell_cls)
        if not dtype:
            continue

        factory_id = None
        for k, v in iter_entries(cfg):
            if k == "factoryID":
                factory_id = v
                break
        if not factory_id:
            continue

        strategy = _strategy_from_factory(factory_id)

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
    """Generate the necessary import statements for the output code."""
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.pmml.missingval.compute.MissingValueHandlerNodeFactory"
)

def _emit_fill_code(settings: MissingValueSettings) -> List[str]:
    """Generate the code to fill missing values based on the provided settings."""
    lines: List[str] = []
    lines.append("out_df = df.copy()")

    ints: List[TypePolicy] = [p for p in settings.by_dtype if p.dtype == "int"]
    floats: List[TypePolicy] = [p for p in settings.by_dtype if p.dtype == "float"]
    strings: List[TypePolicy] = [p for p in settings.by_dtype if p.dtype == "string"]
    booleans: List[TypePolicy] = [p for p in settings.by_dtype if p.dtype == "boolean"]

    def _emit_int(pol: TypePolicy):
        lines.append("int_cols = out_df.select_dtypes(include=['Int64','Int32','Int16','int64','int32','int16']).columns")
        lines.append("if len(int_cols) > 0:")
        if pol.strategy == "fixed":
            if pol.value is not None:
                s = str(pol.value).strip()
                val_literal = s if s.lstrip('+-').isdigit() else f"int({repr(pol.value)})"
                lines.append(f"    out_df[int_cols] = out_df[int_cols].fillna({val_literal}).astype('Int64')")
            else:
                lines.append("    pass  # No fixed value configured for ints; skipping")
        elif pol.strategy in ("mean", "median"):
            fn = "mean" if pol.strategy == "mean" else "median"
            # Per-column: fill with stat if available, round, cast to Int64
            lines.append(f"    out_df[int_cols] = out_df[int_cols].apply(lambda s: (s if pd.isna(s.{fn}()) else s.fillna(s.{fn}()).round()).astype('Int64'))")
        elif pol.strategy == "mode":
            lines.append("    out_df[int_cols] = out_df[int_cols].apply(lambda s: (s.fillna(s.mode().iloc[0]) if not s.mode().empty else s).astype('Int64'))")
        elif pol.strategy == "ffill":
            lines.append("    out_df[int_cols] = out_df[int_cols].apply(lambda s: s.ffill().astype('Int64'))")
        elif pol.strategy == "bfill":
            lines.append("    out_df[int_cols] = out_df[int_cols].apply(lambda s: s.bfill().astype('Int64'))")
        elif pol.strategy == "drop":
            lines.append("    out_df = out_df.dropna(subset=int_cols.tolist())")
        else:
            lines.append(f"    pass  # Unsupported int strategy '{pol.strategy}'")

    def _emit_float(pol: TypePolicy):
        lines.append("float_cols = out_df.select_dtypes(include=['float64','float32']).columns")
        lines.append("if len(float_cols) > 0:")
        if pol.strategy == "fixed":
            if pol.value is not None:
                s = str(pol.value).strip()
                try:
                    float(s)
                    val_literal = s
                except Exception:
                    val_literal = f"float({repr(pol.value)})"
                lines.append(f"    out_df[float_cols] = out_df[float_cols].fillna({val_literal})")
            else:
                lines.append("    pass  # No fixed value configured for floats; skipping")
        elif pol.strategy in ("mean", "median"):
            fn = "mean" if pol.strategy == "mean" else "median"
            lines.append(f"    out_df[float_cols] = out_df[float_cols].apply(lambda s: s.fillna(s.{fn}()))")
        elif pol.strategy == "mode":
            lines.append("    out_df[float_cols] = out_df[float_cols].apply(lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else s))")
        elif pol.strategy == "ffill":
            lines.append("    out_df[float_cols] = out_df[float_cols].ffill()")
        elif pol.strategy == "bfill":
            lines.append("    out_df[float_cols] = out_df[float_cols].bfill()")
        elif pol.strategy == "drop":
            lines.append("    out_df = out_df.dropna(subset=float_cols.tolist())")
        else:
            lines.append(f"    pass  # Unsupported float strategy '{pol.strategy}'")

    def _emit_string(pol: TypePolicy):
        lines.append("str_cols = out_df.select_dtypes(include=['string','object']).columns")
        lines.append("if len(str_cols) > 0:")
        if pol.strategy == "fixed":
            if pol.value is not None:
                lines.append(f"    out_df[str_cols] = out_df[str_cols].fillna({repr(pol.value)})")
            else:
                lines.append("    pass  # No fixed value configured for strings; skipping")
        elif pol.strategy == "mode":
            lines.append("    out_df[str_cols] = out_df[str_cols].apply(lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else s))")
        elif pol.strategy == "ffill":
            lines.append("    out_df[str_cols] = out_df[str_cols].ffill()")
        elif pol.strategy == "bfill":
            lines.append("    out_df[str_cols] = out_df[str_cols].bfill()")
        elif pol.strategy == "drop":
            lines.append("    out_df = out_df.dropna(subset=str_cols.tolist())")
        else:
            lines.append(f"    pass  # Unsupported string strategy '{pol.strategy}'")

    def _emit_boolean(pol: TypePolicy):
        lines.append("bool_cols = out_df.select_dtypes(include=['boolean','bool']).columns")
        lines.append("if len(bool_cols) > 0:")
        if pol.strategy == "fixed":
            if pol.value is not None:
                v = str(pol.value).strip().lower()
                lit = "True" if v in {"true","1","t","y","yes"} else "False"
                lines.append(f"    out_df[bool_cols] = out_df[bool_cols].fillna({lit}).astype('boolean')")
            else:
                lines.append("    pass  # No fixed value configured for booleans; skipping")
        elif pol.strategy == "mode":
            lines.append("    out_df[bool_cols] = out_df[bool_cols].apply(lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else s)).astype('boolean')")
        elif pol.strategy == "ffill":
            lines.append("    out_df[bool_cols] = out_df[bool_cols].ffill().astype('boolean')")
        elif pol.strategy == "bfill":
            lines.append("    out_df[bool_cols] = out_df[bool_cols].bfill().astype('boolean')")
        elif pol.strategy == "drop":
            lines.append("    out_df = out_df.dropna(subset=bool_cols.tolist())")
        else:
            lines.append(f"    pass  # Unsupported boolean strategy '{pol.strategy}'")

    if ints:
        _emit_int(ints[0])
    if floats:
        _emit_float(floats[0])
    if strings:
        _emit_string(strings[0])
    if booleans:
        _emit_boolean(booleans[0])

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
    """Generate the Python code body for the node based on its configuration and input ports."""
    ndir = Path(node_dir) if node_dir else None
    settings = parse_missing_value_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_fill_code(settings))

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
    """Generate the code for a Jupyter notebook cell based on the node's configuration."""
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"

def handle(ntype, nid, npath, incoming, outgoing):
    """Handle the node processing, generating imports and body code."""
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
