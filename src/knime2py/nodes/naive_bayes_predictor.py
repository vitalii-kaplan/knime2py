#!/usr/bin/env python3

"""
Naive Bayes Predictor.

Overview
----------------------------
This module generates Python code for scoring an input table using a Naive Bayes model
bundle produced by the NB Learner. It consumes a model bundle or a bare estimator, writes
a prediction column, and optionally appends per-class probability columns. The scored
table is output to the node's context.

Runtime Behavior
----------------------------
Inputs:
- Reads a model bundle (dict with {'estimator', 'features', 'target', 'classes', 'meta': {...}})
  or a bare sklearn estimator from context.
- Reads a data table to score from context.

Outputs:
- Writes the scored DataFrame to context with the key mapping to the output port.
- The prediction column is named based on settings, defaulting to "Prediction (<target>)".
- Optionally, adds probability columns for each class.

Key algorithms or mappings:
- Aligns features based on the model bundle or estimator attributes.
- Handles missing columns by adding them with zero values and dropping extra columns.

Edge Cases
----------------------------
The code implements safeguards for:
- Empty or constant columns.
- NaN values in the input data.
- Class imbalance by providing fallback paths for predictions.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries:
- pandas
These dependencies are required by the generated code, not by this module.

Usage
----------------------------
This module is typically invoked by the knime2py emitter, which generates the necessary
Python code for KNIME nodes. An example of expected context access is:
```
context['model_key'] = model_bundle
context['data_key'] = data_table
```

Node Identity
----------------------------
KNIME factory id:
- FACTORY = "org.knime.base.node.mine.bayes.naivebayes.predictor4.NaiveBayesPredictorNodeFactory3"

Configuration
----------------------------
The settings are defined in the `NaiveBayesPredictorSettings` dataclass with the following
important fields:
- change_prediction: bool = True (determines if the prediction column name can be changed)
- pred_col_name: Optional[str] = None (custom name for the prediction column)
- include_probs: bool = True (determines if probability columns are included)
- prob_suffix: str = "_NB" (suffix for probability column names)

The `parse_nb_predictor_settings` function extracts these values from the settings.xml file
using XPath queries and provides fallbacks when necessary.

Limitations
----------------------------
This module does not support certain options available in KNIME, such as advanced
parameter tuning or specific model configurations.

References
----------------------------
For more information, refer to the KNIME documentation and the following hub URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.node.mine.bayes.naivebayes.predictor4.NaiveBayesPredictorNodeFactory3
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
FACTORY = "org.knime.base.node.mine.bayes.naivebayes.predictor4.NaiveBayesPredictorNodeFactory3"

# ---------------------------------------------------------------------
# settings.xml → NaiveBayesPredictorSettings
# ---------------------------------------------------------------------

@dataclass
class NaiveBayesPredictorSettings:
    change_prediction: bool = True
    pred_col_name: Optional[str] = None
    include_probs: bool = True
    prob_suffix: str = "_NB"

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

def parse_nb_predictor_settings(node_dir: Optional[Path]) -> NaiveBayesPredictorSettings:
    """
    Parse the Naive Bayes predictor settings from the settings.xml file.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        NaiveBayesPredictorSettings: The parsed settings.
    """
    if not node_dir:
        return NaiveBayesPredictorSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return NaiveBayesPredictorSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return NaiveBayesPredictorSettings()

    change_pred = _bool(first(model_el, ".//*[local-name()='entry' and @key='change prediction']/@value"), True)
    pred_name = first(model_el, ".//*[local-name()='entry' and @key='prediction column name']/@value")
    incl_probs = _bool(first(model_el, ".//*[local-name()='entry' and @key='inclProbVals']/@value"), True)
    prob_suf = first(model_el, ".//*[local-name()='entry' and @key='class probability suffix']/@value") or "_NB"

    return NaiveBayesPredictorSettings(
        change_prediction=change_pred,
        pred_col_name=(pred_name or None),
        include_probs=incl_probs,
        prob_suffix=prob_suf,
    )

# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    """
    Generate the necessary imports for the Naive Bayes predictor.

    Returns:
        List[str]: A list of import statements.
    """
    # Only pandas needed; estimator comes from the Learner bundle
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.bayes.naivebayes.predictor4.NaiveBayesPredictorNodeFactory3"
)

def _emit_predict_code(cfg: NaiveBayesPredictorSettings) -> List[str]:
    """
    Generate the prediction code based on the provided configuration.

    Args:
        cfg (NaiveBayesPredictorSettings): The settings for the Naive Bayes predictor.

    Returns:
        List[str]: The lines of code for making predictions.
    """
    lines: List[str] = []
    lines.append("# Fetch inputs")
    lines.append("model_obj = context[model_key]")
    lines.append("df_in = context[data_key]")
    lines.append("out_df = df_in.copy()")
    lines.append("")
    lines.append("# Unify to a dict-like bundle with expected keys")
    lines.append("if isinstance(model_obj, dict):")
    lines.append("    bundle = model_obj")
    lines.append("else:")
    lines.append("    bundle = {'estimator': model_obj}")  # fallback: bare estimator
    lines.append("")
    lines.append("est = bundle.get('estimator') or bundle.get('model')")
    lines.append("feat = bundle.get('features') or bundle.get('feature_cols') or getattr(est, 'feature_names_in_', None)")
    lines.append("tgt  = bundle.get('target') or bundle.get('y_col') or bundle.get('target_name')")
    lines.append("classes = list(bundle.get('classes') or getattr(est, 'classes_', []))")
    lines.append("meta = bundle.get('meta', {}) if isinstance(bundle.get('meta'), dict) else {}")
    lines.append("skip_missing = bool(meta.get('skip_missing', False))")
    lines.append("")
    lines.append("# Normalize features to a plain list (avoid NumPy ambiguity)")
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
    # Rebuild X
    lines.append("# --- Rebuild feature matrix X ---")
    lines.append("cand_cols = [c for c in out_df.columns if c != tgt]")
    lines.append("num_cols = out_df[cand_cols].select_dtypes(include=['number','bool','boolean','Int64','Float64']).columns.tolist()")
    lines.append("non_num_cols = [c for c in cand_cols if c not in set(num_cols)]")
    lines.append("X_num = out_df[num_cols].apply(pd.to_numeric, errors='coerce')")
    lines.append("if skip_missing:")
    lines.append("    X_num = X_num")
    lines.append("else:")
    lines.append("    X_num = X_num.apply(lambda s: s.fillna(s.mean()))")
    lines.append("X_cat = pd.get_dummies(out_df[non_num_cols], dummy_na=(not skip_missing)) if non_num_cols else pd.DataFrame(index=out_df.index)")
    lines.append("X_full = pd.concat([X_num, X_cat], axis=1)")
    lines.append("")
    lines.append("# If we know the expected features, align exactly to them")
    lines.append("if feat_list:")
    lines.append("    for c in feat_list:")
    lines.append("        if c not in X_full.columns:")
    lines.append("            X_full[c] = 0.0")
    lines.append("    X = X_full[feat_list]")
    lines.append("else:")
    lines.append("    # No feature list available; use what we built (best-effort)")
    lines.append("    X = X_full")
    lines.append("")
    # Prediction column name
    if cfg.change_prediction and cfg.pred_col_name:
        lines.append(f"pred_col = {repr(cfg.pred_col_name)}")
    else:
        # default to "Prediction (target)"
        lines.append("pred_col = f'Prediction ({tgt or \"target\"})'")
    lines.append("")
    lines.append("# Predict labels")
    lines.append("pred = est.predict(X)")
    lines.append("out_df[pred_col] = pd.Series(pred, index=out_df.index).astype('object')")
    lines.append("")
    # Probabilities
    if cfg.include_probs:
        lines.append("# Probabilities per class (if supported)")
        lines.append("if hasattr(est, 'predict_proba'):")
        lines.append("    proba = est.predict_proba(X)")
        lines.append(f"    _suf = {repr(cfg.prob_suffix)}")
        lines.append("    if not classes and getattr(proba, 'shape', (0, 0))[1] == 2:")
        lines.append("        classes = ['class0', 'class1']")
        lines.append("    for j, cls in enumerate(classes):")
        lines.append("        cname = f\"P ({tgt}={cls}){_suf}\"")
        lines.append("        out_df[cname] = proba[:, j]")
    else:
        lines.append("# Probabilities disabled by settings")
    lines.append("")
    lines.append("# Publish")
    lines.append("context[out_port_key] = out_df")
    return lines

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],   # Port 1 = model bundle, Port 2 = data
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the body of the Python code for the Naive Bayes predictor.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: The lines of code for the node's body.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_nb_predictor_settings(ndir)

    # Map inputs: ensure port 1 -> model, port 2 -> data (prefer target_port if provided)
    pairs = normalize_in_ports(in_ports)
    model_src, model_in = pairs[0] if pairs else ("UNKNOWN", "1")
    data_src, data_in = (pairs[1] if len(pairs) > 1 else ("UNKNOWN", "2"))

    ports = out_ports or ["1"]
    out_port = sorted({(p or "1") for p in ports})[0]

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"model_key = '{model_src}:{model_in}'")
    lines.append(f"data_key = '{data_src}:{data_in}'")
    lines.append(f"out_port_key = '{node_id}:{out_port}'")
    lines.extend(_emit_predict_code(cfg))
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
        str: The generated code for the notebook cell.
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

    # Prefer target-port mapping when available
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
