#!/usr/bin/env python3

"""
Utilities for building PMML documents used by generated code.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET

PMML_NS = "http://www.dmg.org/PMML-4_4"

__all__ = [
    "build_pmml_document",
    "pmml_to_string",
    "build_normalizer_pmml",
    "emit_normalizer_pmml_builder",
    "emit_missing_value_bundle_builder",
]


def build_pmml_document(
    *,
    version: str = "4.4",
    application: str = "knime2py",
    app_version: str = "1.0",
) -> ET.Element:
    """Create a PMML root element with a header."""
    root = ET.Element("PMML", version=version, xmlns=PMML_NS)
    header = ET.SubElement(root, "Header")
    ET.SubElement(header, "Application", name=application, version=app_version)
    return root


def pmml_to_string(root: ET.Element) -> str:
    """Serialize an ElementTree root to a unicode string with XML declaration."""
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8")


def _ensure_transformation_dict(root: ET.Element) -> ET.Element:
    trans = root.find("./TransformationDictionary")
    if trans is None:
        trans = ET.SubElement(root, "TransformationDictionary")
    return trans


def build_normalizer_pmml(
    columns: Iterable[str],
    stats: Dict[str, Dict[str, Optional[float]]],
    mode: str,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> str:
    """
    Build a PMML TransformationDictionary describing normalization of columns.
    """
    cols = [c for c in columns if c]
    root = build_pmml_document()
    if not cols:
        return pmml_to_string(root)

    trans = _ensure_transformation_dict(root)
    upper_mode = (mode or "MINMAX").upper()

    for col in cols:
        info = stats.get(col) or {}
        if upper_mode == "MINMAX":
            mn = info.get("min")
            mx = info.get("max")
            if mn is None or mx is None:
                continue
            derived = ET.SubElement(
                trans,
                "DerivedField",
                name=f"{col}_norm",
                optype="continuous",
                dataType="double",
            )
            norm = ET.SubElement(derived, "NormContinuous", origField=str(col))
            ET.SubElement(norm, "LinearNorm", orig=str(mn), norm=str(new_min))
            ET.SubElement(norm, "LinearNorm", orig=str(mx), norm=str(new_max))
        else:
            mu = info.get("mean")
            sd = info.get("std")
            if sd in (None, 0):
                continue
            derived = ET.SubElement(
                trans,
                "DerivedField",
                name=f"{col}_z",
                optype="continuous",
                dataType="double",
            )
            apply_div = ET.SubElement(derived, "Apply", function="/")
            apply_sub = ET.SubElement(apply_div, "Apply", function="-")
            ET.SubElement(apply_sub, "FieldRef", field=str(col))
            const_mu = ET.SubElement(apply_sub, "Constant", dataType="double")
            const_mu.text = "0.0" if mu is None else str(mu)
            const_sd = ET.SubElement(apply_div, "Constant", dataType="double")
            const_sd.text = str(sd)

    return pmml_to_string(root)


def emit_normalizer_pmml_builder(fn_name: str = "_build_normalizer_pmml") -> List[str]:
    """
    Return Python source lines that define a runtime helper for building Normalizer PMML.
    The emitted code only depends on xml.etree.ElementTree.
    """
    lines = [
        f"def {fn_name}(columns, stats, mode, new_min, new_max):",
        "    cols = [c for c in (columns or []) if c]",
        "    root = ET.Element('PMML', version='4.4', xmlns='http://www.dmg.org/PMML-4_4')",
        "    header = ET.SubElement(root, 'Header')",
        "    ET.SubElement(header, 'Application', name='knime2py', version='1.0')",
        "    if not cols:",
        "        return ET.tostring(root, encoding='utf-8', xml_declaration=True).decode('utf-8')",
        "    trans = ET.SubElement(root, 'TransformationDictionary')",
        "    upper_mode = (mode or 'MINMAX').upper()",
        "    for col in cols:",
        "        info = stats.get(col) or {}",
        "        if upper_mode == 'MINMAX':",
        "            mn = info.get('min')",
        "            mx = info.get('max')",
        "            if mn is None or mx is None:",
        "                continue",
        "            derived = ET.SubElement(",
        "                trans,",
        "                'DerivedField',",
        "                name=f\"{col}_norm\",",
        "                optype='continuous',",
        "                dataType='double',",
        "            )",
        "            norm = ET.SubElement(derived, 'NormContinuous', origField=str(col))",
        "            ET.SubElement(norm, 'LinearNorm', orig=str(mn), norm=str(new_min))",
        "            ET.SubElement(norm, 'LinearNorm', orig=str(mx), norm=str(new_max))",
        "        else:",
        "            mu = info.get('mean')",
        "            sd = info.get('std')",
        "            if sd in (None, 0):",
        "                continue",
        "            derived = ET.SubElement(",
        "                trans,",
        "                'DerivedField',",
        "                name=f\"{col}_z\",",
        "                optype='continuous',",
        "                dataType='double',",
        "            )",
        "            apply_div = ET.SubElement(derived, 'Apply', function='/')",
        "            apply_sub = ET.SubElement(apply_div, 'Apply', function='-')",
        "            ET.SubElement(apply_sub, 'FieldRef', field=str(col))",
        "            const_mu = ET.SubElement(apply_sub, 'Constant', dataType='double')",
        "            const_mu.text = '0.0' if mu is None else str(mu)",
        "            const_sd = ET.SubElement(apply_div, 'Constant', dataType='double')",
        "            const_sd.text = str(sd)",
        "    return ET.tostring(root, encoding='utf-8', xml_declaration=True).decode('utf-8')",
    ]
    return lines


def emit_missing_value_bundle_builder(
    dtype_fn: str = "_mv_dtype_key",
    policy_fn: str = "_mv_policy_for",
    resolve_fn: str = "_mv_resolve_value",
    bundle_fn: str = "_mv_collect_bundle",
) -> List[str]:
    """Return helper functions used by Missing Value nodes to build bundle metadata."""
    return [
        f"def {dtype_fn}(series):",
        "    import pandas as pd",
        "    if pd.api.types.is_integer_dtype(series):",
        "        return 'int'",
        "    if pd.api.types.is_float_dtype(series):",
        "        return 'float'",
        "    if pd.api.types.is_bool_dtype(series):",
        "        return 'boolean'",
        "    return 'string'",
        "",
        f"def {policy_fn}(col_name, dtype_key, column_policies, dtype_policies):",
        "    for pol in column_policies:",
        "        if pol.get('column') == col_name:",
        "            return pol",
        "    for pol in dtype_policies:",
        "        if pol.get('dtype') == dtype_key:",
        "            return pol",
        "    return None",
        "",
        f"def {resolve_fn}(series, policy, dtype_key):",
        "    import pandas as pd",
        "    if policy is None:",
        "        return None",
        "    strat = (policy.get('strategy') or '').lower()",
        "    if strat == 'fixed':",
        "        val = policy.get('value')",
        "    elif strat == 'mean':",
        "        val = series.mean(skipna=True)",
        "    elif strat == 'median':",
        "        val = series.median(skipna=True)",
        "    elif strat == 'mode':",
        "        mode = series.mode(dropna=True)",
        "        val = mode.iloc[0] if not mode.empty else None",
        "    else:",
        "        val = policy.get('value')",
        "    if val is None or (isinstance(val, float) and pd.isna(val)):",
        "        return None",
        "    if dtype_key == 'int':",
        "        try:",
        "            val = int(round(float(val)))",
        "        except Exception:",
        "            return None",
        "        return str(val), 'integer'",
        "    if dtype_key == 'float':",
        "        try:",
        "            val = float(val)",
        "        except Exception:",
        "            return None",
        "        return str(val), 'double'",
        "    if dtype_key == 'boolean':",
        "        text = str(val).strip().lower()",
        "        val_int = '1' if text in {'true','1','t','y','yes'} else '0'",
        "        return val_int, 'integer'",
        "    return str(val), 'string'",
        "",
        f"def {bundle_fn}(source_df, column_policies, dtype_policies):",
        "    df = source_df",
        "    data_entries = []",
        "    transforms = []",
        "    for col in df.columns:",
        "        series = df[col]",
        f"        dtype_key = {dtype_fn}(series)",
        "        entry = {'name': str(col), 'dtype_key': dtype_key}",
        "        if dtype_key == 'string':",
        "            entry['optype'] = 'categorical'",
        "            entry['dataType'] = 'string'",
        "            values = series.dropna().astype(str).unique().tolist()",
        "            if values:",
        "                entry['values'] = [str(v) for v in values[:128]]",
        "        else:",
        "            entry['optype'] = 'continuous'",
        "            entry['dataType'] = 'integer' if dtype_key in {'int','boolean'} else 'double'",
        "            non_missing = series.dropna()",
        "            if not non_missing.empty:",
        "                try:",
        "                    entry['interval'] = [float(non_missing.min()), float(non_missing.max())]",
        "                except Exception:",
        "                    pass",
        "        data_entries.append(entry)",
        f"        policy = {policy_fn}(col, dtype_key, column_policies, dtype_policies)",
        f"        resolved = {resolve_fn}(series, policy, dtype_key)",
        "        if resolved is None:",
        "            continue",
        "        const_text, const_dtype = resolved",
        "        transforms.append({",
        "            'column': str(col),",
        "            'derived_name': f\"{col}*\",",
        "            'const': const_text,",
        "            'const_dtype': const_dtype,",
        "            'optype': entry.get('optype', 'continuous'),",
        "        })",
        "    return {",
        "        'model_type': 'missing_value',",
        "        'version': '4.2',",
        "        'application': {'name': 'knime2py', 'version': '1.0'},",
        "        'data_dictionary': data_entries,",
        "        'transformations': transforms,",
        "    }",
    ]
