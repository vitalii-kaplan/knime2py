#!/usr/bin/env python3

"""
SVM Predictor.

Overview
----------------------------
Scores an input table using an SVC estimator produced by the SVM Learner. This module
emits Python code that consumes a model bundle, writes a prediction column, and optionally
appends per-class probability columns, outputting the scored table to the context.

Runtime Behavior
----------------------------
Inputs:
- Reads a model bundle from context using the key format 'model_src:model_in'.
- Reads a DataFrame from context using the key format 'data_src:data_in'.

Outputs:
- Writes the scored DataFrame to context with the key format 'node_id:port', where
  port defaults to '1'.

Key algorithms or mappings:
- The module handles feature selection, ensuring that the features used for prediction
  are derived from the model bundle or the input DataFrame. It also manages class
  probabilities if available.

Edge Cases
----------------------------
The code implements safeguards for empty inputs, creating empty prediction and probability
columns when the input DataFrame has zero rows. It also handles missing feature columns
gracefully.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries: pandas. These dependencies
are required for the generated code, not for this module.

Usage
----------------------------
Typically, this module is invoked by the knime2py emitter, which generates code for
the SVM predictor node. An example of expected context access is:
```
context['model_key'] = model_bundle
context['data_key'] = input_dataframe
```

Node Identity
----------------------------
KNIME factory id: org.knime.base.node.mine.svm.predictor2.SVMPredictorNodeFactory.

Configuration
----------------------------
The settings are defined in the `PredictorSettings` dataclass, which includes:
- change_prediction: bool (default: False) - Indicates if the prediction column name should
  be changed.
- pred_name: Optional[str] (default: None) - Custom name for the prediction column.
- add_probabilities: bool (default: True) - Indicates if probability columns should be added.
- prob_suffix: str (default: "_SV") - Suffix for probability column names.

The `parse_predictor_settings` function extracts these values from the settings.xml file.

Limitations
----------------------------
This module does not support all features of KNIME's SVM Predictor, such as advanced
parameter tuning or specific model types.

References
----------------------------
For more information, refer to the KNIME documentation and the hub link:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/org.knime.base.node.mine.svm.predictor2.SVMPredictorNodeFactory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (
    first,
    first_el,
    normalize_in_ports,
    collect_module_imports,
    split_out_imports,
)

# KNIME factory (from settings.xml)
FACTORY = "org.knime.base.node.mine.svm.predictor2.SVMPredictorNodeFactory"

# Hub link (reference)
HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.svm.predictor2.SVMPredictorNodeFactory"
)


# ---------------------------------------------------------------------
# settings.xml → PredictorSettings
# ---------------------------------------------------------------------

@dataclass
class PredictorSettings:
    change_prediction: bool = False
    pred_name: Optional[str] = None
    add_probabilities: bool = True
    prob_suffix: str = "_SV"


def _bool(s: Optional[str], default: bool) -> bool:
    """
    Convert a string to a boolean value.

    Args:
        s (Optional[str]): The string to convert.
        default (bool): The default value to return if s is None.

    Returns:
        bool: The converted boolean value.
    """
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def parse_predictor_settings(node_dir: Optional[Path]) -> PredictorSettings:
    """
    Parse the predictor settings from the settings.xml file.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        PredictorSettings: The parsed predictor settings.
    """
    if not node_dir:
        return PredictorSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return PredictorSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return PredictorSettings()

    # Keys per the SVM Predictor XML
    pred_name = first(model_el, ".//*[local-name()='entry' and @key='prediction column name']/@value")
    change_pred = _bool(first(model_el, ".//*[local-name()='entry' and @key='change prediction']/@value"), False)

    add_probs = _bool(first(model_el, ".//*[local-name()='entry' and @key='add probabilities']/@value"), True)
    prob_suffix = first(model_el, ".//*[local-name()='entry' and @key='class probability suffix']/@value") or "_SV"

    return PredictorSettings(
        change_prediction=change_pred,
        pred_name=(pred_name or None),
        add_probabilities=add_probs,
        prob_suffix=prob_suffix,
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    """
    Generate the necessary import statements for the code.

    Returns:
        List[str]: A list of import statements.
    """
    # Only pandas needed here; estimator comes from the Learner bundle
    return ["import pandas as pd"]


def _emit_predict_code(cfg: PredictorSettings) -> List[str]:
    """
    Generate the prediction code based on the provided predictor settings.

    Args:
        cfg (PredictorSettings): The settings for the predictor.

    Returns:
        List[str]: The lines of code for making predictions.
    """
    lines: List[str] = []
    lines.append("model_obj = context[model_key]")
    lines.append("df = context[data_key]")
    lines.append("out_df = df.copy()")
    lines.append("")
    lines.append("# Normalize to a dict-like bundle with expected keys")
    lines.append("if isinstance(model_obj, dict):")
    lines.append("    bundle = model_obj")
    lines.append("else:")
    lines.append("    # Fallback: wrap bare estimator")
    lines.append("    bundle = {'estimator': model_obj}")
    lines.append("")
    lines.append("est = bundle.get('estimator') or bundle.get('model')")
    lines.append("if est is None:")
    lines.append("    raise ValueError('SVM Predictor: missing estimator in model bundle')")
    lines.append("feat = bundle.get('features') or getattr(est, 'feature_names_in_', None)")
    lines.append("tgt  = bundle.get('target') or bundle.get('y_col') or bundle.get('target_name')")
    lines.append("classes = list(bundle.get('classes') or getattr(est, 'classes_', []))")
    lines.append("")
    lines.append("# Normalize features to a plain list")
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
    lines.append("if len(feat_list) == 0:")
    lines.append("    if tgt and tgt in out_df.columns:")
    lines.append("        feat_list = [c for c in out_df.columns if c != tgt]")
    lines.append("    else:")
    lines.append("        feat_list = list(out_df.columns)")
    lines.append("")
    lines.append("missing = [c for c in feat_list if c not in out_df.columns]")
    lines.append("if missing:")
    lines.append("    raise KeyError(f'SVM Predictor: missing feature columns in data: {missing}')")
    lines.append("X = out_df[feat_list]")
    lines.append("")
    # Prediction column name
    if cfg.change_prediction and (cfg.pred_name or "").strip():
        lines.append(f"pred_col = {repr(cfg.pred_name)}")
    else:
        lines.append("pred_col = f'Prediction ({tgt or \"target\"})'")
    lines.append(f"_suf = {repr(cfg.prob_suffix)}")
    lines.append("")
    # === EMPTY-INPUT GUARD ===
    lines.append("if X.shape[0] == 0:")
    lines.append("    # Create empty prediction/probability columns and publish.")
    lines.append("    out_df[pred_col] = pd.Series(index=out_df.index, dtype='object')")
    lines.append("    if bool(" + ("True" if cfg.add_probabilities else "False") + "):")
    lines.append("        _cls = classes or list(getattr(est, 'classes_', []))")
    lines.append("        for cls in _cls:")
    lines.append("            cname = f\"P ({tgt}={cls}){_suf}\"")
    lines.append("            out_df[cname] = pd.Series(index=out_df.index, dtype='float64')")
    lines.append("else:")
    lines.append("    # Normal scoring path")
    lines.append("    pred = est.predict(X)")
    lines.append("    out_df[pred_col] = pd.Series(pred, index=out_df.index).astype('object')")
    lines.append("    if bool(" + ("True" if cfg.add_probabilities else "False") + ") and hasattr(est, 'predict_proba'):")
    lines.append("        try:")
    lines.append("            proba = est.predict_proba(X)")
    lines.append("        except Exception:")
    lines.append("            proba = None")
    lines.append("        if proba is not None:")
    lines.append("            # Align column order with estimator.classes_ when available")
    lines.append("            _cls_order = list(getattr(est, 'classes_', [])) or list(classes)")
    lines.append("            for j, cls in enumerate(_cls_order):")
    lines.append("                cname = f\"P ({tgt}={cls}){_suf}\"")
    lines.append("                out_df[cname] = proba[:, j]")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],   # Port 1 = model bundle, Port 2 = data
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the body of the Python code for the SVM predictor node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: The lines of code for the node's body.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_predictor_settings(ndir)

    # Map inputs: ensure port 1 -> model, port 2 -> data (fallback if not explicitly wired)
    pairs = normalize_in_ports(in_ports)
    model_src, model_in = pairs[0] if pairs else ("UNKNOWN", "1")
    data_src, data_in = (pairs[1] if len(pairs) > 1 else ("UNKNOWN", "2"))

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"model_key = '{model_src}:{model_in}'")
    lines.append(f"data_key = '{data_src}:{data_in}'")

    lines.extend(_emit_predict_code(cfg))

    # Single output port: predicted table
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
    """
    Generate the code for a Jupyter notebook cell for the SVM predictor node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        str: The generated code as a string.
    """
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handle the processing of the SVM predictor node.

    Returns (imports, body_lines) if this module can handle the node; else None.

    We map inputs so that:
      - **Port 1** (this node's target port 1) → model bundle key
      - **Port 2** (this node's target port 2) → data frame key

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming connections.
        outgoing: The outgoing connections.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines.
    """
    explicit_imports = collect_module_imports(generate_imports)

    # Determine upstream keys for our input ports
    model_pair: Optional[Tuple[str, str]] = None
    data_pair: Optional[Tuple[str, str]] = None
    for src_id, e in (incoming or []):
        src_port = str(getattr(e, "source_port", "") or "1")
        tgt_port = str(getattr(e, "target_port", "") or "")
        if tgt_port == "1":
            model_pair = (str(src_id), src_port)
        elif tgt_port == "2":
            data_pair = (str(src_id), src_port)

    norm_in: List[Tuple[str, str]] = []
    if model_pair:
        norm_in.append(model_pair)
    if data_pair:
        norm_in.append(data_pair)
    if not norm_in:
        # Fallback to whatever ordering is present
        norm_in = [(str(src), str(getattr(e, "source_port", "") or "1")) for src, e in (incoming or [])]

    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, norm_in, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
