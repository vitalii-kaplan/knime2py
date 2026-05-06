#!/usr/bin/env python3

"""Linear Regression Learner node."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import (
    collect_module_imports,
    first,
    first_el,
    iter_entries,
    normalize_in_ports,
    split_out_imports,
)


FACTORY = "org.knime.base.node.mine.regression.linear2.learner.LinReg2LearnerNodeFactory2"


@dataclass
class LinearRegressionSettings:
    target: Optional[str] = None
    include_cols: List[str] = field(default_factory=list)
    exclude_cols: List[str] = field(default_factory=list)
    include_constant: bool = True
    missing_value_handling: str = "fail"


def _bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _collect_numeric_name_entries(cfg: ET._Element) -> List[str]:
    out: List[str] = []
    for key, value in iter_entries(cfg):
        if key.isdigit() and value:
            out.append(value.strip())
    return out


def parse_linear_regression_settings(node_dir: Optional[Path]) -> LinearRegressionSettings:
    if not node_dir:
        return LinearRegressionSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return LinearRegressionSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return LinearRegressionSettings()

    target = first(model_el, ".//*[local-name()='entry' and @key='target']/@value")
    include_constant = _bool(
        first(model_el, ".//*[local-name()='entry' and @key='include_constant']/@value"),
        True,
    )
    missing_value_handling = (
        first(model_el, ".//*[local-name()='entry' and @key='missing_value_handling']/@value")
        or "fail"
    )

    include_cols: List[str] = []
    exclude_cols: List[str] = []
    cf = first_el(model_el, ".//*[local-name()='config' and @key='column_filter']")
    if cf is not None:
        inc_cfg = first_el(cf, ".//*[local-name()='config' and @key='included_names']")
        exc_cfg = first_el(cf, ".//*[local-name()='config' and @key='excluded_names']")
        if inc_cfg is not None:
            include_cols.extend(_collect_numeric_name_entries(inc_cfg))
        if exc_cfg is not None:
            exclude_cols.extend(_collect_numeric_name_entries(exc_cfg))

    return LinearRegressionSettings(
        target=target or None,
        include_cols=list(dict.fromkeys(include_cols)),
        exclude_cols=list(dict.fromkeys(exclude_cols)),
        include_constant=include_constant,
        missing_value_handling=missing_value_handling,
    )


def generate_imports() -> List[str]:
    return [
        "import pandas as pd",
        "import numpy as np",
        "from scipy import stats as _scipy_stats",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.regression.linear2.learner.LinReg2LearnerNodeFactory2"
)


def _emit_train_code(cfg: LinearRegressionSettings) -> List[str]:
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    if not cfg.target:
        lines.append("model_bundle = {'estimator': None, 'features': [], 'target': None}")
        lines.append("coef_df = pd.DataFrame(columns=['Variable', 'Coeff.', 'Std. Err.', 't-value', 'P>|t|'])")
        lines.append("summary_df = pd.DataFrame([{'error': 'no target configured'}])")
        return lines

    lines.append(f"_target = {repr(cfg.target)}")
    lines.append("if _target not in df.columns:")
    lines.append("    raise KeyError(f'Linear Regression Learner: target column not found: {_target!r}')")

    if cfg.include_cols:
        lines.append(f"_include_cols = {repr(cfg.include_cols)}")
        lines.append("source_cols = [c for c in _include_cols if c in df.columns and c != _target]")
    else:
        lines.append("source_cols = [c for c in df.columns if c != _target]")

    if cfg.exclude_cols:
        lines.append(f"_exclude_cols = {repr(cfg.exclude_cols)}")
        lines.append("source_cols = [c for c in source_cols if c not in set(_exclude_cols)]")

    lines.append("if not source_cols:")
    lines.append("    raise ValueError('Linear Regression Learner: no feature columns selected')")
    lines.append(f"_include_constant = {bool(cfg.include_constant)!r}")
    lines.append(f"_missing_mode = {repr((cfg.missing_value_handling or 'fail').lower())}")
    lines.append("")
    lines.append("feature_info = []")
    lines.append("x_parts = []")
    lines.append("for col in source_cols:")
    lines.append("    s = df[col]")
    lines.append("    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):")
    lines.append("        vals = pd.to_numeric(s, errors='coerce').astype(float)")
    lines.append("        x_parts.append(vals.to_frame(col))")
    lines.append("        feature_info.append({'kind': 'numeric', 'column': col, 'features': [col]})")
    lines.append("    else:")
    lines.append("        cat = s.astype('object').where(s.notna(), 'Missing').astype(str)")
    lines.append("        dummies = pd.get_dummies(cat, prefix=col, prefix_sep='=', dtype=float)")
    lines.append("        drop_level = None")
    lines.append("        if len(dummies.columns) > 0:")
    lines.append("            drop_col = dummies.columns[0]")
    lines.append("            drop_level = str(drop_col).split('=', 1)[1] if '=' in str(drop_col) else str(drop_col)")
    lines.append("            dummies = dummies.iloc[:, 1:]")
    lines.append("        if len(dummies.columns) > 0:")
    lines.append("            x_parts.append(dummies)")
    lines.append("        feature_info.append({")
    lines.append("            'kind': 'categorical',")
    lines.append("            'column': col,")
    lines.append("            'levels': sorted(cat.dropna().unique().tolist()),")
    lines.append("            'drop_level': drop_level,")
    lines.append("            'features': list(dummies.columns),")
    lines.append("        })")
    lines.append("")
    lines.append("X_df = pd.concat(x_parts, axis=1) if x_parts else pd.DataFrame(index=df.index)")
    lines.append("X_df = X_df.astype(float)")
    lines.append("y = pd.to_numeric(df[_target], errors='coerce').astype(float)")
    lines.append("valid_mask = y.notna() & ~X_df.isna().any(axis=1)")
    lines.append("if _missing_mode == 'fail' and not bool(valid_mask.all()):")
    lines.append("    raise ValueError('Linear Regression Learner: missing values found in target or features')")
    lines.append("X_fit = X_df.loc[valid_mask].copy()")
    lines.append("y_fit = y.loc[valid_mask].copy()")
    lines.append("if X_fit.empty:")
    lines.append("    raise ValueError('Linear Regression Learner: no training rows available')")
    lines.append("")
    lines.append("design_cols = list(X_fit.columns)")
    lines.append("X_mat = X_fit.to_numpy(dtype=float)")
    lines.append("if _include_constant:")
    lines.append("    X_mat = np.column_stack([X_mat, np.ones(len(X_fit), dtype=float)])")
    lines.append("    design_cols_with_intercept = design_cols + ['Intercept']")
    lines.append("else:")
    lines.append("    design_cols_with_intercept = design_cols")
    lines.append("y_vec = y_fit.to_numpy(dtype=float)")
    lines.append("coef = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]")
    lines.append("pred_fit = X_mat @ coef")
    lines.append("resid = y_vec - pred_fit")
    lines.append("df_resid = int(max(len(y_vec) - X_mat.shape[1], 0))")
    lines.append("if df_resid > 0:")
    lines.append("    sigma2 = float((resid @ resid) / df_resid)")
    lines.append("    cov = sigma2 * np.linalg.pinv(X_mat.T @ X_mat)")
    lines.append("    std_err = np.sqrt(np.diag(cov))")
    lines.append("    t_vals = np.divide(coef, std_err, out=np.full_like(coef, np.nan, dtype=float), where=std_err != 0)")
    lines.append("    p_vals = 2 * _scipy_stats.t.sf(np.abs(t_vals), df_resid)")
    lines.append("else:")
    lines.append("    std_err = np.full_like(coef, np.nan, dtype=float)")
    lines.append("    t_vals = np.full_like(coef, np.nan, dtype=float)")
    lines.append("    p_vals = np.full_like(coef, np.nan, dtype=float)")
    lines.append("coef_df = pd.DataFrame({")
    lines.append("    'Variable': design_cols_with_intercept,")
    lines.append("    'Coeff.': coef,")
    lines.append("    'Std. Err.': std_err,")
    lines.append("    't-value': t_vals,")
    lines.append("    'P>|t|': p_vals,")
    lines.append("})")
    lines.append("summary_df = pd.DataFrame([{'n_rows': len(y_vec), 'n_features': len(design_cols), 'df_resid': df_resid, 'r_squared': float(1 - (resid @ resid) / np.sum((y_vec - y_vec.mean()) ** 2)) if len(y_vec) > 1 and np.sum((y_vec - y_vec.mean()) ** 2) != 0 else np.nan}])")
    lines.append("model_bundle = {")
    lines.append("    'kind': 'linear_regression',")
    lines.append("    'target': _target,")
    lines.append("    'source_cols': list(source_cols),")
    lines.append("    'features': list(design_cols),")
    lines.append("    'feature_cols': list(design_cols),")
    lines.append("    'feature_info': feature_info,")
    lines.append("    'include_constant': _include_constant,")
    lines.append("    'coef': coef.tolist(),")
    lines.append("    'coefficients': coef.tolist(),")
    lines.append("    'intercept': float(coef[-1]) if _include_constant and len(coef) else 0.0,")
    lines.append("}")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_linear_regression_settings(ndir)

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0] if pairs else ("UNKNOWN", "1")

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")
    lines.extend(_emit_train_code(cfg))

    ports = [str(p or "1") for p in (out_ports or ["1", "2"])]
    port_map = {"1": "model_bundle", "2": "coef_df", "3": "summary_df"}
    for p in sorted(set(ports), key=lambda value: (0, int(value)) if value.isdigit() else (1, value)):
        lines.append(f"context['{node_id}:{p}'] = {port_map.get(p, 'model_bundle')}")

    return lines


def get_name() -> str:
    return "Linear Regression Learner"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
