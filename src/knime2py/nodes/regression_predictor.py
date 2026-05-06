#!/usr/bin/env python3

"""Regression Predictor node."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, normalize_in_ports, split_out_imports


FACTORY = "org.knime.base.node.mine.regression.predict3.RegressionPredictorNodeFactory2"


@dataclass
class RegressionPredictorSettings:
    has_custom_name: bool = False
    custom_name: Optional[str] = None


def _bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_regression_predictor_settings(node_dir: Optional[Path]) -> RegressionPredictorSettings:
    if not node_dir:
        return RegressionPredictorSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return RegressionPredictorSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return RegressionPredictorSettings()

    # KNIME writes the misspelled key name in settings.xml.
    has_custom = _bool(
        first(model_el, ".//*[local-name()='entry' and @key='has_custom_predicition_name']/@value"),
        False,
    )
    custom_name = first(model_el, ".//*[local-name()='entry' and @key='custom_prediction_name']/@value")
    return RegressionPredictorSettings(has_custom_name=has_custom, custom_name=custom_name or None)


def generate_imports() -> List[str]:
    return ["import pandas as pd", "import numpy as np"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.regression.predict3.RegressionPredictorNodeFactory2"
)


def _emit_predict_code(cfg: RegressionPredictorSettings) -> List[str]:
    lines: List[str] = []
    lines.append("model_obj = context[model_key]")
    lines.append("df = context[data_key]")
    lines.append("out_df = df.copy()")
    lines.append("bundle = model_obj if isinstance(model_obj, dict) else {'estimator': model_obj}")
    lines.append("target = bundle.get('target') or bundle.get('target_name') or bundle.get('y_col')")
    if cfg.has_custom_name and cfg.custom_name:
        lines.append(f"pred_col = {repr(cfg.custom_name)}")
    else:
        lines.append("pred_col = f'Prediction ({target or \"target\"})'")
    lines.append("")
    lines.append("if bundle.get('kind') == 'linear_regression' or 'coef' in bundle or 'coefficients' in bundle:")
    lines.append("    feature_info = list(bundle.get('feature_info') or [])")
    lines.append("    x_parts = []")
    lines.append("    for info in feature_info:")
    lines.append("        col = info.get('column')")
    lines.append("        if col not in out_df.columns:")
    lines.append("            raise KeyError(f'Regression Predictor: missing feature column: {col!r}')")
    lines.append("        if info.get('kind') == 'numeric':")
    lines.append("            feat_name = (info.get('features') or [col])[0]")
    lines.append("            vals = pd.to_numeric(out_df[col], errors='coerce').astype(float)")
    lines.append("            x_parts.append(vals.to_frame(feat_name))")
    lines.append("        else:")
    lines.append("            cat = out_df[col].astype('object').where(out_df[col].notna(), 'Missing').astype(str)")
    lines.append("            for feat_name in info.get('features') or []:")
    lines.append("                level = str(feat_name).split('=', 1)[1] if '=' in str(feat_name) else str(feat_name)")
    lines.append("                x_parts.append((cat == level).astype(float).to_frame(feat_name))")
    lines.append("    X_df = pd.concat(x_parts, axis=1) if x_parts else pd.DataFrame(index=out_df.index)")
    lines.append("    feature_cols = list(bundle.get('features') or bundle.get('feature_cols') or X_df.columns)")
    lines.append("    for col in feature_cols:")
    lines.append("        if col not in X_df.columns:")
    lines.append("            X_df[col] = 0.0")
    lines.append("    X_df = X_df[feature_cols].astype(float)")
    lines.append("    coef = np.asarray(bundle.get('coef') or bundle.get('coefficients'), dtype=float)")
    lines.append("    if bool(bundle.get('include_constant', True)):")
    lines.append("        if len(coef) != len(feature_cols) + 1:")
    lines.append("            raise ValueError('Regression Predictor: coefficient count does not match feature count')")
    lines.append("        pred = X_df.to_numpy(dtype=float) @ coef[:-1] + coef[-1]")
    lines.append("    else:")
    lines.append("        if len(coef) != len(feature_cols):")
    lines.append("            raise ValueError('Regression Predictor: coefficient count does not match feature count')")
    lines.append("        pred = X_df.to_numpy(dtype=float) @ coef")
    lines.append("else:")
    lines.append("    est = bundle.get('estimator') or bundle.get('model')")
    lines.append("    if est is None:")
    lines.append("        raise ValueError('Regression Predictor: missing estimator/model bundle')")
    lines.append("    feature_cols = list(bundle.get('features') or bundle.get('feature_cols') or getattr(est, 'feature_names_in_', []))")
    lines.append("    if not feature_cols:")
    lines.append("        feature_cols = [c for c in out_df.columns if c != target]")
    lines.append("    missing = [c for c in feature_cols if c not in out_df.columns]")
    lines.append("    if missing:")
    lines.append("        raise KeyError(f'Regression Predictor: missing feature columns: {missing}')")
    lines.append("    pred = est.predict(out_df[feature_cols])")
    lines.append("out_df[pred_col] = pd.Series(pred, index=out_df.index).astype(float)")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_regression_predictor_settings(ndir)

    pairs = normalize_in_ports(in_ports)
    model_src, model_in = pairs[0] if pairs else ("UNKNOWN", "1")
    data_src, data_in = pairs[1] if len(pairs) > 1 else ("UNKNOWN", "2")

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"model_key = '{model_src}:{model_in}'")
    lines.append(f"data_key = '{data_src}:{data_in}'")
    lines.extend(_emit_predict_code(cfg))

    for p in sorted({str(p or '1') for p in (out_ports or ['1'])}):
        lines.append(f"context['{node_id}:{p}'] = out_df")
    return lines


def get_name() -> str:
    return "Regression Predictor"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)

    model_pair: Optional[Tuple[str, str]] = None
    data_pair: Optional[Tuple[str, str]] = None
    for src_id, edge in incoming or []:
        src_port = str(getattr(edge, "source_port", "") or "1")
        tgt_port = str(getattr(edge, "target_port", "") or "")
        if tgt_port == "1":
            model_pair = (str(src_id), src_port)
        elif tgt_port == "2":
            data_pair = (str(src_id), src_port)

    in_ports: List[Tuple[str, str]] = []
    if model_pair:
        in_ports.append(model_pair)
    if data_pair:
        in_ports.append(data_pair)
    if not in_ports:
        in_ports = [(str(src), str(getattr(edge, "source_port", "") or "1")) for src, edge in (incoming or [])]

    out_ports = [str(getattr(edge, "source_port", "") or "1") for _, edge in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
