#!/usr/bin/env python3

"""
Logistic Regression Predictor.

Overview
----------------------------
This module generates Python code to score an input table using a LogisticRegression
estimator produced by the LR Learner. It consumes a model bundle, writes a prediction
column, and optionally appends per-class probability columns, outputting the scored
table to the node's context.

Runtime Behavior
----------------------------
Inputs:
- Reads a model bundle and input data from the context using specified keys.

Outputs:
- Writes the scored DataFrame to context with the key format `context['<node_id>:<port>']`.

Key algorithms or mappings:
- The module handles model loading, feature extraction, and prediction using the
  provided estimator. It includes logic for fallback feature selection and
  probability column generation.

Edge Cases
----------------------------
The code implements safeguards against missing feature columns, empty DataFrames,
and class imbalance. It provides fallback paths for feature extraction and handles
cases where the model may not provide expected attributes.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries: pandas, numpy,
sklearn, imblearn, matplotlib, lxml. These dependencies are required for the
functionality of the generated code, not this module.

Usage
----------------------------
This module is typically invoked by the knime2py emitter, which generates code
for KNIME nodes. An example of expected context access is:
```
context['model_key'] = model_bundle
context['data_key'] = input_data
```

Node Identity
----------------------------
KNIME factory id:
- FACTORY = "org.knime.base.node.mine.regression.logistic.predictor.LogisticRegressionPredictorNodeFactory"

Configuration
----------------------------
The settings are encapsulated in the `PredictorSettings` dataclass, which includes:
- `has_custom_name`: Indicates if a custom prediction column name is used (default: False).
- `custom_name`: The custom name for the prediction column (default: None).
- `include_probs`: Flag to include probability columns (default: True).
- `prob_suffix`: Suffix for probability column names (default: "_LR").

The `parse_predictor_settings` function extracts these values from the settings.xml
file using XPath queries, with fallbacks for missing entries.

Limitations
----------------------------
This module does not support certain advanced configurations available in KNIME,
such as custom handling of class imbalance or specific model types not covered
by the standard Logistic Regression implementation.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.node.mine.regression.logistic.predictor.LogisticRegressionPredictorNodeFactory
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

# KNIME factory
FACTORY = "org.knime.base.node.mine.regression.logistic.predictor.LogisticRegressionPredictorNodeFactory"

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
        PredictorSettings: An instance of PredictorSettings with the parsed values.
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

    # Note: KNIME uses these misspellings in XML and we must read them verbatim.
    has_custom = _bool(
        first(model_el, ".//*[local-name()='entry' and @key='has_custom_predicition_name']/@value"),
        False,
    )
    custom_name = first(
        model_el, ".//*[local-name()='entry' and @key='custom_prediction_name']/@value"
    )

    include_probs = _bool(
        first(model_el, ".//*[local-name()='entry' and @key='include_probabilites']/@value"),
        True,
    )
    prob_suffix = (
        first(model_el, ".//*[local-name()='entry' and @key='propability_columns_suffix']/@value")
        or "_LR"
    )

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
    """
    Generate the necessary import statements for the generated code.

    Returns:
        List[str]: A list of import statements.
    """
    # Only pandas needed here; estimator comes from the Learner bundle
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.regression.logistic.predictor.LogisticRegressionPredictorNodeFactory"
)

def _emit_predict_code(cfg: PredictorSettings) -> List[str]:
    """
    Generate the prediction code based on the provided predictor settings.

    Args:
        cfg (PredictorSettings): The settings for the predictor.

    Returns:
        List[str]: A list of code lines for making predictions.
    """
    lines: List[str] = []
    lines.append("model_obj = context[model_key]")
    lines.append("df = context[data_key]")
    lines.append("out_df = df.copy()")
    lines.append("")
    lines.append("# Unify to a dict-like bundle with expected keys")
    lines.append("if isinstance(model_obj, dict):")
    lines.append("    bundle = model_obj")
    lines.append("else:")
    lines.append("    # Fallback: wrap bare estimator")
    lines.append("    bundle = {'estimator': model_obj}")
    lines.append("")
    lines.append("est = bundle.get('estimator') or bundle.get('model')")
    lines.append("feat = bundle.get('features') or getattr(est, 'feature_names_in_', None)")
    lines.append("tgt  = bundle.get('target') or bundle.get('y_col') or bundle.get('target_name')")
    lines.append("classes = list(bundle.get('classes') or getattr(est, 'classes_', []))")
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
    lines.append("if len(feat_list) == 0:")
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
        lines.append("    if not classes and getattr(proba, 'shape', (0, 0))[1] == 2:")
        lines.append("        classes = ['class0', 'class1']")
        lines.append(f"    _suf = {repr(cfg.prob_suffix)}")
        lines.append("    for j, cls in enumerate(classes):")
        lines.append("        cname = f\"P ({tgt}={cls}){_suf}\"")
        lines.append("        out_df[cname] = proba[:, j]")
    else:
        lines.append("# Probabilities disabled by settings")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],   # Port 1 = model bundle, Port 2 = data
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python code body for the node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: A list of code lines for the node's functionality.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_predictor_settings(ndir)

    # Map inputs: ensure port 1 -> model, port 2 -> data
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
    Generate the code for a Jupyter notebook cell.

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
    Handle the node and return the necessary imports and body lines.

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
        norm_in = [(str(src), str(getattr(e, "source_port", "") or "1")) for src, e in (incoming or [])]

    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, norm_in, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
