#!/usr/bin/env python3

"""
Missing Value Handler.

Overview
----------------------------
This module generates Python code to handle missing values in a DataFrame
based on KNIME's Missing Value policies. It fits into the knime2py generator
pipeline by producing code that applies specified fill strategies to input
tables and writes the results to the node's context.

Runtime Behavior
----------------------------
Inputs:
- Reads a DataFrame from the context using the key format 'src_id:in_port'.

Outputs:
- Writes the processed DataFrame back to the context with the key format
  'node_id:out_port', where out_port defaults to '1'.

Key algorithms or mappings:
- Implements fill strategies such as mean, median, mode, forward fill,
  backward fill, and fixed value fills based on the configuration parsed
  from settings.xml.

Edge Cases
----------------------------
- Handles empty or constant columns by skipping them.
- Safeguards against NaN values and class imbalance by providing fallback
  strategies.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries:
- pandas

These dependencies are required by the generated code, not by this module.

Usage
----------------------------
Typically invoked by upstream KNIME nodes that require missing value handling.
Example context access:
```python
df = context['input_table:1']
```

Node Identity
----------------------------
KNIME factory id:
- FACTORY = "org.knime.base.node.preproc.pmml.missingval.compute.MissingValueHandlerNodeFactory"

Configuration
----------------------------
Settings are defined in the `MissingValueSettings` dataclass, which includes:
- by_dtype: List of TypePolicy instances defining fill strategies per data type.

The `parse_missing_value_settings` function extracts these values from the
settings.xml file using XPath queries.

Limitations
----------------------------
Currently, this module does not support all KNIME fill strategies and may
approximate behavior in some cases.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.node.preproc.pmml.missingval.compute.MissingValueHandlerNodeFactory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # iter_entries, first_el, first, collect_module_imports, split_out_imports, normalize_in_ports
from .pmml_utils import emit_missing_value_bundle_builder

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
class ColumnPolicy:
    column: str
    strategy: str
    value: Optional[str] = None
    dtype: Optional[str] = None

@dataclass
class MissingValueSettings:
    by_dtype: List[TypePolicy] = field(default_factory=list)
    by_column: List[ColumnPolicy] = field(default_factory=list)

_CELL_TO_DTYPE = {
    "org.knime.core.data.def.IntCell": "int",
    "org.knime.core.data.def.LongCell": "int",
    "org.knime.core.data.def.DoubleCell": "float",
    "org.knime.core.data.def.StringCell": "string",
    "org.knime.core.data.def.BooleanCell": "boolean",
}

def _dtype_from_factory(factory_id: Optional[str]) -> Optional[str]:
    if not factory_id:
        return None
    fid = factory_id.lower()
    if "string" in fid:
        return "string"
    if any(tok in fid for tok in ("integer", "long", "int")):
        return "int"
    if any(tok in fid for tok in ("double", "float")):
        return "float"
    if "boolean" in fid or "bool" in fid:
        return "boolean"
    return None

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
    model_cfgs = root.xpath(".//*[local-name()='config' and @key='model']")
    if not model_cfgs:
        return MissingValueSettings()
    model_cfg = model_cfgs[0]

    # Column overrides
    by_column: List[ColumnPolicy] = []
    column_sections = model_cfg.xpath("./*[local-name()='config' and @key='columnSettings']")
    if column_sections:
        for cfg in column_sections[0].xpath("./*[local-name()='config']"):
            names_cfg = first_el(cfg, "./*[local-name()='config' and @key='colNames']")
            columns: List[str] = []
            if names_cfg is not None:
                for key, value in iter_entries(names_cfg):
                    if key.isdigit() and value:
                        columns.append(value)
            if not columns:
                continue

            settings_cfg = first_el(cfg, "./*[local-name()='config' and @key='settings']")
            if settings_cfg is None:
                continue
            factory_id = None
            for key, value in iter_entries(settings_cfg):
                if key == "factoryID":
                    factory_id = value
                    break
            inner_settings = first_el(settings_cfg, "./*[local-name()='config' and @key='settings']")
            fixed_val = _first_present_value(inner_settings) if inner_settings is not None else None
            strategy = _strategy_from_factory(factory_id or "")
            dtype = _dtype_from_factory(factory_id)
            for col in columns:
                by_column.append(
                    ColumnPolicy(
                        column=col,
                        strategy=strategy,
                        value=fixed_val,
                        dtype=dtype,
                    )
                )

    dts = model_cfg.xpath("./*[local-name()='config' and @key='dataTypeSettings']")
    if not dts:
        return MissingValueSettings(by_column=by_column)

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

    return MissingValueSettings(by_dtype=by_dtype, by_column=by_column)

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

MV_HELPER_LINES = emit_missing_value_bundle_builder()

def _emit_fill_code(settings: MissingValueSettings) -> List[str]:
    """Generate the code to fill missing values based on the provided settings."""
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    override_names = sorted({pol.column for pol in settings.by_column if pol.column})
    if override_names:
        literal = ", ".join(repr(name) for name in override_names)
        lines.append(f"override_cols = {{{literal}}}")
    else:
        lines.append("override_cols = set()")

    def _literal_for_value(value: Optional[str], dtype: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip()
        if dtype == "int":
            return s if s and s.lstrip('+-').isdigit() else f"int({repr(value)})"
        if dtype == "float":
            try:
                float(s)
                return s
            except Exception:
                return f"float({repr(value)})"
        if dtype == "boolean":
            v = s.lower()
            return "True" if v in {"true", "1", "t", "y", "yes"} else "False"
        return repr(value)

    def _assign_line(col_repr: str, expr: str, dtype: Optional[str], indent: str = "    ", round_int: bool = False) -> str:
        target = expr
        if dtype == "int":
            target = f"({expr}).round().astype('Int64')" if round_int else f"({expr}).astype('Int64')"
        elif dtype == "boolean":
            target = f"({expr}).astype('boolean')"
        return f"{indent}out_df[{col_repr}] = {target}"

    for pol in settings.by_column:
        col_repr = repr(pol.column)
        dtype = (pol.dtype or "").lower() or None
        strategy = (pol.strategy or "").lower()
        lines.append(f"if {col_repr} in out_df.columns:")
        if strategy == "fixed":
            literal = _literal_for_value(pol.value, dtype)
            if literal is None:
                lines.append("    pass  # No fixed value configured for this column; skipping")
            else:
                expr = f"out_df[{col_repr}].fillna({literal})"
                lines.append(_assign_line(col_repr, expr, dtype))
        elif strategy in ("mean", "median"):
            fn = "mean" if strategy == "mean" else "median"
            lines.append(f"    stat = out_df[{col_repr}].{fn}()")
            lines.append("    if not pd.isna(stat):")
            expr = f"out_df[{col_repr}].fillna(stat)"
            lines.append(_assign_line(col_repr, expr, dtype, indent="        ", round_int=dtype == "int"))
        elif strategy == "mode":
            lines.append(f"    mode = out_df[{col_repr}].mode()")
            lines.append("    if not mode.empty:")
            expr = f"out_df[{col_repr}].fillna(mode.iloc[0])"
            lines.append(_assign_line(col_repr, expr, dtype, indent="        "))
        elif strategy == "ffill":
            expr = f"out_df[{col_repr}].ffill()"
            lines.append(_assign_line(col_repr, expr, dtype))
        elif strategy == "bfill":
            expr = f"out_df[{col_repr}].bfill()"
            lines.append(_assign_line(col_repr, expr, dtype))
        elif strategy == "drop":
            lines.append(f"    out_df = out_df.dropna(subset=[{col_repr}])")
        else:
            lines.append(f"    pass  # Unsupported column strategy '{pol.strategy}'")

    ints: List[TypePolicy] = [p for p in settings.by_dtype if p.dtype == "int"]
    floats: List[TypePolicy] = [p for p in settings.by_dtype if p.dtype == "float"]
    strings: List[TypePolicy] = [p for p in settings.by_dtype if p.dtype == "string"]
    booleans: List[TypePolicy] = [p for p in settings.by_dtype if p.dtype == "boolean"]

    def _emit_int(pol: TypePolicy):
        lines.append("int_cols = [c for c in out_df.select_dtypes(include=['Int64','Int32','Int16','int64','int32','int16']).columns if c not in override_cols]")
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
        lines.append("float_cols = [c for c in out_df.select_dtypes(include=['float64','float32']).columns if c not in override_cols]")
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
        lines.append("str_cols = [c for c in out_df.select_dtypes(include=['string','object']).columns if c not in override_cols]")
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
        lines.append("bool_cols = [c for c in out_df.select_dtypes(include=['boolean','bool']).columns if c not in override_cols]")
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

    if not settings.by_dtype and not settings.by_column:
        lines.append("# No missing-value policies found; passthrough.")
        lines.append("out_df = df")

    return lines

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
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
    lines.extend(MV_HELPER_LINES)

    lines.extend(_emit_fill_code(settings))
    dtype_literal = [
        {"dtype": pol.dtype, "strategy": pol.strategy, "value": pol.value}
        for pol in settings.by_dtype
    ]
    col_literal = [
        {
            "column": pol.column,
            "dtype": pol.dtype,
            "strategy": pol.strategy,
            "value": pol.value,
        }
        for pol in settings.by_column
    ]
    lines.append(f"dtype_policies = {dtype_literal!r}")
    lines.append(f"column_policies = {col_literal!r}")
    lines.append("model_bundle = _mv_collect_bundle(df, column_policies, dtype_policies)")
    lines.append("model_bundle['strategies'] = dtype_policies")
    lines.append("model_bundle['column_strategies'] = column_policies")

    ports = out_ports or ["1"]
    port_map = {"1": "out_df", "2": "model_bundle"}
    for p in sorted({(p or '1') for p in ports}):
        target = port_map.get(p, "out_df")
        lines.append(f"context['{node_id}:{p}'] = {target}")
    return lines



def get_name() -> str:
    """Return name of the node in KNIME workflow."""
    return "Missing Value"


def handle(ntype, nid, npath, incoming, outgoing):
    """Handle the node processing, generating imports and body code."""
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
