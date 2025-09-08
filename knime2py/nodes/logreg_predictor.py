#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (
    first,
    first_el,
    iter_entries,
    normalize_in_ports,
    collect_module_imports,
    split_out_imports,
)

# KNIME factory
PREDICTOR_FACTORY = (
    "org.knime.base.node.mine.regression.logistic.predictor.LogisticRegressionPredictorNodeFactory"
)

def can_handle(node_type: Optional[str]) -> bool:
    return bool(node_type and node_type.endswith(PREDICTOR_FACTORY))


# ---------------------------------------------------------------------
# settings.xml → PredictorSettings
# ---------------------------------------------------------------------

@dataclass
class PredictorSettings:
    has_custom_name: bool = False
    custom_name: Optional[str] = None
    include_probs: bool = True
    prob_suffix: str = "_LR"


def _bool(s: Optional[str], default: bool) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def parse_predictor_settings(node_dir: Optional[Path]) -> PredictorSettings:
    if not node_dir:
        return PredictorSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return PredictorSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return PredictorSettings()

    has_custom = _bool(first(model_el, ".//*[local-name()='entry' and @key='has_custom_predicition_name']/@value"), False)
    # Note: KNIME uses that exact misspelling "predicition" in the XML.
    custom_name = first(model_el, ".//*[local-name()='entry' and @key='custom_prediction_name']/@value")
    include_probs = _bool(first(model_el, ".//*[local-name()='entry' and @key='include_probabilites']/@value"), True)
    prob_suffix = first(model_el, ".//*[local-name()='entry' and @key='propability_columns_suffix']/@value") or "_LR"

    return PredictorSettings(
        has_custom_name=has_custom,
        custom_name=(custom_name or None),
        include_probs=include_probs,
        prob_suffix=prob_suffix,
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    # Only pandas needed; we call predict/predict_proba on the sklearn estimator stored by the Learner.
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.regression.logistic.predictor.LogisticRegressionPredictorNodeFactory"
)

def _emit_predict_code(cfg: PredictorSettings) -> List[str]:
    """
    Accept either:
      - a dict-like bundle with keys: estimator/model, features/feature_cols, target, classes, or
      - a bare sklearn estimator (e.g., LogisticRegression).
    """
    lines: List[str] = []
    lines.append("model_obj = context[model_key]")
    lines.append("df = context[data_key]")
    lines.append("out_df = df.copy()")
    lines.append("")
    lines.append("# Handle dict-like bundle vs bare estimator")
    lines.append("if isinstance(model_obj, dict):")
    lines.append("    est = model_obj.get('estimator') or model_obj.get('model') or model_obj")
    lines.append("    feat = (model_obj.get('features') or "
                 "            model_obj.get('feature_cols') or "
                 "            getattr(est, 'feature_names_in_', None))")
    lines.append("    tgt  = (model_obj.get('target') or model_obj.get('y_col') or "
                 "            model_obj.get('target_name'))")
    lines.append("    classes = list(model_obj.get('classes') or getattr(est, 'classes_', []))")
    lines.append("else:")
    lines.append("    est = model_obj")
    lines.append("    feat = getattr(est, 'feature_names_in_', None)")
    lines.append("    tgt  = None")
    lines.append("    classes = list(getattr(est, 'classes_', []))")
    lines.append("")
    lines.append("# Normalize features to a plain list (avoid NumPy truth-value ambiguity)")
    lines.append("if isinstance(feat, (list, tuple)):")
    lines.append("    feat_list = list(feat)")
    lines.append("elif getattr(feat, 'tolist', None):")
    lines.append("    feat_list = list(feat.tolist())")
    lines.append("elif feat is None:")
    lines.append("    feat_list = []")
    lines.append("else:")
    lines.append("    try:")
    lines.append("        feat_list = list(feat)")
    lines.append("    except Exception:")
    lines.append("        feat_list = []")
    lines.append("")
    lines.append("# Feature fallback if none provided by bundle/estimator")
    lines.append("if not feat_list:")
    lines.append("    if tgt and tgt in out_df.columns:")
    lines.append("        feat_list = [c for c in out_df.columns if c != tgt]")
    lines.append("    else:")
    lines.append("        feat_list = list(out_df.columns)")
    lines.append("")
    lines.append("missing = [c for c in feat_list if c not in out_df.columns]")
    lines.append("if missing:")
    lines.append("    raise KeyError(f'Missing feature columns in data: {missing}')")
    lines.append("X = out_df[feat_list]")
    lines.append("")
    # Prediction column name
    if cfg.has_custom_name:
        lines.append(f"pred_col = {repr(cfg.custom_name)}")
    else:
        lines.append("pred_col = f'Prediction ({tgt or \"target\"})'")
    lines.append("pred = est.predict(X)")
    lines.append("out_df[pred_col] = pd.Series(pred, index=out_df.index).astype('object')")
    lines.append("")
    # Probabilities (optional)
    if cfg.include_probs:
        lines.append("# Probability columns per class (if supported)")
        lines.append("if hasattr(est, 'predict_proba'):")
        lines.append("    proba = est.predict_proba(X)")
        lines.append("    if not classes and proba.shape[1] == 2:")
        lines.append("        classes = ['class0', 'class1']")
        lines.append(f"    _suf = {repr(cfg.prob_suffix)}")
        lines.append("    for j, cls in enumerate(classes):")
        lines.append("        cname = f\"P({cls}){_suf}\"")
        lines.append("        out_df[cname] = proba[:, j]")
    else:
        lines.append("# Probabilities disabled by settings")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],   # we will receive two inputs: model (port 1), data (port 2)
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_predictor_settings(ndir)

    # We want (model_key, data_key) in that order. Our handle() will pass them ordered, but keep a fallback.
    pairs = normalize_in_ports(in_ports)
    # Fallbacks if caller passed just two pairs without target-port info
    model_src, model_in = pairs[0] if pairs else ("UNKNOWN", "1")
    data_src, data_in = (pairs[1] if len(pairs) > 1 else ("UNKNOWN", "2"))

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"model_key = '{model_src}:{model_in}'")
    lines.append(f"data_key = '{data_src}:{data_in}'")

    lines.extend(_emit_predict_code(cfg))

    # Publish to context (default single output port '1')
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
    Returns (imports, body_lines) if this module can handle the node; else None.

    We map inputs so that:
      - **Port 1** (this node's target port 1) → model bundle key
      - **Port 2** (this node's target port 2) → data frame key
    """
    if not (ntype and can_handle(ntype)):
        return None

    # explicit imports declared by this node module
    explicit_imports = collect_module_imports(generate_imports)

    # Determine upstream context keys for each of our target ports
    model_pair: Optional[Tuple[str, str]] = None
    data_pair: Optional[Tuple[str, str]] = None
    for src_id, e in (incoming or []):
        src_port = str(getattr(e, "source_port", "") or "1")      # upstream's source port
        tgt_port = str(getattr(e, "target_port", "") or "")       # this node's input port
        if tgt_port == "1":
            model_pair = (str(src_id), src_port)
        elif tgt_port == "2":
            data_pair = (str(src_id), src_port)

    # Fallbacks if target ports were missing/odd: preserve original order
    norm_in = []
    if model_pair:
        norm_in.append(model_pair)
    if data_pair:
        norm_in.append(data_pair)
    if not norm_in:
        # Last resort: just convert incoming to (src, source_port) pairs in given order
        norm_in = [(str(src), str(getattr(e, "source_port", "") or "1")) for src, e in (incoming or [])]

    # One output (table with predictions)
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, norm_in, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
