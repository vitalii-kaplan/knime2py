#!/usr/bin/env python3

"""Naive Bayes Learner module.

Overview
----------------------------
This module generates Python code for a Naive Bayes learner using scikit-learn,
based on configurations from a KNIME node's settings.xml. It produces a bundle
for downstream predictions and a training summary DataFrame.

Runtime Behavior
----------------------------
Inputs:
- Reads a training DataFrame from the context using the specified input port.

Outputs:
- Writes a bundle dict to context['{node_id}:1'] containing the trained model,
  features, target, classes, and metadata.
- Writes a summary DataFrame to context['{node_id}:2'] with training details.

Key algorithms or mappings:
- Utilizes GaussianNB from scikit-learn for classification.
- Handles numeric and categorical features, applying one-hot encoding for limited
  cardinality categorical columns.

Edge Cases
----------------------------
The code implements safeguards against:
- Missing target columns, raising KeyErrors if not found.
- Empty or constant feature columns, raising ValueErrors if no usable features
  are available.
- NaN values in numeric features, with options to drop or impute them based on
  configuration.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries:
- pandas
- numpy
- sklearn

These dependencies are required by the generated code, not by this module.

Usage
----------------------------
Typically invoked by a KNIME workflow, this module can be used in conjunction
with nodes that provide training data and expect a trained model and summary
as output. Example context access:
```python
df = context['{src_id}:{in_port}']  # training table
```

Node Identity
----------------------------
KNIME factory id:
- FACTORY = "org.knime.base.node.mine.bayes.naivebayes.learner3.NaiveBayesLearnerNodeFactory4"

Configuration
----------------------------
The settings are encapsulated in the `NaiveBayesSettings` dataclass, which includes:
- target: The target column for classification (default: None).
- var_smoothing: Smoothing parameter for GaussianNB (default: 1e-9).
- min_sd_value: Minimum standard deviation value (default: 1e-4).
- min_sd_threshold: Minimum standard deviation threshold (default: 0.0).
- max_nominal_vals: Maximum number of unique values for categorical columns (default: 20).
- skip_missing: Whether to drop rows with missing values (default: False).

The `parse_nb_settings` function extracts these values from the settings.xml file
using XPath queries, with fallbacks to default values.

Limitations
----------------------------
Certain KNIME behaviors may not be fully supported, such as advanced handling
of class imbalance or specific imputation strategies.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.node.mine.bayes.naivebayes.learner3.NaiveBayesLearnerNodeFactory4
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (  # first, first_el, normalize_in_ports, collect_module_imports, split_out_imports
    first,
    first_el,
    normalize_in_ports,
    collect_module_imports,
    split_out_imports,
)

# KNIME factory (Naive Bayes Learner v4)
FACTORY = "org.knime.base.node.mine.bayes.naivebayes.learner3.NaiveBayesLearnerNodeFactory4"

# -------------------------------------------------------------------------------------------------
# settings.xml → NaiveBayesSettings
# -------------------------------------------------------------------------------------------------

@dataclass
class NaiveBayesSettings:
    target: Optional[str] = None
    var_smoothing: float = 1e-9      # maps from "threshold"
    min_sd_value: float = 1e-4       # informational
    min_sd_threshold: float = 0.0    # informational
    max_nominal_vals: int = 20
    skip_missing: bool = False
    compatible_pmml: bool = False    # informational

def _to_int(s: Optional[str], default: int) -> int:
    """Convert a string to an integer, returning a default value if conversion fails."""
    try:
        return int(str(s))
    except Exception:
        return default

def _to_float(s: Optional[str], default: float) -> float:
    """Convert a string to a float, returning a default value if conversion fails."""
    try:
        return float(str(s))
    except Exception:
        return default

def _to_bool(s: Optional[str], default: bool) -> bool:
    """Convert a string to a boolean, returning a default value if conversion fails."""
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}

def parse_nb_settings(node_dir: Optional[Path]) -> NaiveBayesSettings:
    """Parse the settings.xml file to extract Naive Bayes settings."""
    if not node_dir:
        return NaiveBayesSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return NaiveBayesSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return NaiveBayesSettings()

    target = first(model_el, ".//*[local-name()='entry' and @key='classifyColumn']/@value")
    var_smoothing = _to_float(first(model_el, ".//*[local-name()='entry' and @key='threshold']/@value"), 1e-9)
    min_sd_value = _to_float(first(model_el, ".//*[local-name()='entry' and @key='minSdValue']/@value"), 1e-4)
    min_sd_threshold = _to_float(first(model_el, ".//*[local-name()='entry' and @key='minSdThreshold']/@value"), 0.0)
    max_nominal_vals = _to_int(first(model_el, ".//*[local-name()='entry' and @key='maxNoOfNomVals']/@value"), 20)
    skip_missing = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='skipMissingVals']/@value"), False)
    compat_pmml = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='compatiblePMML']/@value"), False)

    # guardrails
    if var_smoothing <= 0.0:
        var_smoothing = 1e-9
    if max_nominal_vals < 1:
        max_nominal_vals = 1

    return NaiveBayesSettings(
        target=target or None,
        var_smoothing=float(var_smoothing),
        min_sd_value=float(min_sd_value),
        min_sd_threshold=float(min_sd_threshold),
        max_nominal_vals=int(max_nominal_vals),
        skip_missing=bool(skip_missing),
        compatible_pmml=bool(compat_pmml),
    )

# -------------------------------------------------------------------------------------------------
# Code generators
# -------------------------------------------------------------------------------------------------

def generate_imports():
    """Generate the necessary import statements for the Naive Bayes learner."""
    return [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.naive_bayes import GaussianNB",
    ]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.bayes.naivebayes.learner3.NaiveBayesLearnerNodeFactory4"
)

def _emit_train_code(cfg: NaiveBayesSettings) -> List[str]:
    """
    Emit code that:
      - builds X from numeric + limited-cardinality categoricals (one-hot),
      - handles missing values per setting,
      - fits GaussianNB(var_smoothing),
      - produces a bundle + summary_df.
    """
    lines: List[str] = []

    if not cfg.target:
        lines += [
            "df_in = df.copy()",
            "# No target configured; emit empty bundle/summary.",
            "bundle = {'estimator': None, 'features': [], 'target': None, 'classes': [], 'meta': {'error': 'no target configured'}}",
            "summary_df = pd.DataFrame([{'error': 'no target configured'}])",
        ]
        return lines

    # Persist settings to runtime vars
    lines.append(f"_target = {repr(cfg.target)}")
    lines.append(f"_var_smoothing = float({cfg.var_smoothing})")
    lines.append(f"_max_nom = int({cfg.max_nominal_vals})")
    lines.append(f"_skip_missing = {('True' if cfg.skip_missing else 'False')}")
    lines.append(f"_min_sd_value = float({cfg.min_sd_value})   # informational")
    lines.append(f"_min_sd_threshold = float({cfg.min_sd_threshold})  # informational")
    lines.append("df_in = df.copy()")
    lines.append("")
    lines.append("# Validate target")
    lines.append("if _target not in df_in.columns:")
    lines.append("    raise KeyError(f'NaiveBayes: target column not found: {_target!r}')")
    lines.append("")
    lines.append("# Separate features/target")
    lines.append("y = df_in[_target]")
    lines.append("feat_candidates = [c for c in df_in.columns if c != _target]")
    lines.append("")
    lines.append("# Split numeric vs non-numeric")
    lines.append("num_cols = df_in[feat_candidates].select_dtypes(include=['number','bool','boolean','Int64','Float64']).columns.tolist()")
    lines.append("non_num_cols = [c for c in feat_candidates if c not in set(num_cols)]")
    lines.append("")
    lines.append("# Limit categoricals by cardinality (≤ _max_nom)")
    lines.append("cat_cols = []")
    lines.append("for c in non_num_cols:")
    lines.append("    try:")
    lines.append("        nun = int(df_in[c].nunique(dropna=not _skip_missing))")
    lines.append("    except Exception:")
    lines.append("        nun = _max_nom + 1")
    lines.append("    if nun <= _max_nom:")
    lines.append("        cat_cols.append(c)")
    lines.append("")
    lines.append("# Build numeric matrix")
    lines.append("X_num = df_in[num_cols].apply(pd.to_numeric, errors='coerce')")
    lines.append("if _skip_missing:")
    lines.append("    # drop rows with any NaN in numeric features")
    lines.append("    X_num = X_num")
    lines.append("else:")
    lines.append("    # impute numeric NaNs with column mean")
    lines.append("    X_num = X_num.apply(lambda s: s.fillna(s.mean()))")
    lines.append("")
    lines.append("# One-hot encode approved categoricals")
    lines.append("if cat_cols:")
    lines.append("    X_cat = pd.get_dummies(df_in[cat_cols], dummy_na=(not _skip_missing))")
    lines.append("else:")
    lines.append("    X_cat = pd.DataFrame(index=df_in.index)")
    lines.append("")
    lines.append("# Concatenate features")
    lines.append("X = pd.concat([X_num, X_cat], axis=1)")
    lines.append("")
    lines.append("# Align y with X (apply row filter per _skip_missing policy)")
    lines.append("if _skip_missing:")
    lines.append("    mask = ~X.isna().any(axis=1)")
    lines.append("    X = X.loc[mask]")
    lines.append("    y = y.loc[mask]")
    lines.append("")
    lines.append("if X.shape[1] == 0:")
    lines.append("    raise ValueError('NaiveBayes: no usable feature columns found')")
    lines.append("")
    lines.append("# Train GaussianNB")
    lines.append("nb = GaussianNB(var_smoothing=_var_smoothing)")
    lines.append("nb.fit(X, y)")
    lines.append("")
    lines.append("# Bundle + summary")
    lines.append("feat_cols = X.columns.tolist()")
    lines.append("classes = list(getattr(nb, 'classes_', []))")
    lines.append("bundle = {")
    lines.append("    'estimator': nb,")
    lines.append("    'model': nb,")  # alias
    lines.append("    'features': list(feat_cols),")
    lines.append("    'feature_cols': list(feat_cols),")  # alias
    lines.append("    'target': _target,")
    lines.append("    'classes': classes,")
    lines.append("    'meta': {")
    lines.append("        'var_smoothing': _var_smoothing,")
    lines.append("        'n_numeric': len(num_cols),")
    lines.append("        'n_categorical_used': len(cat_cols),")
    lines.append("        'skip_missing': _skip_missing,")
    lines.append("        'minSdValue': _min_sd_value,")
    lines.append("        'minSdThreshold': _min_sd_threshold,")
    lines.append("        'maxNoOfNomVals': _max_nom,")
    lines.append("    },")
    lines.append("}")
    lines.append("")
    lines.append("summary_df = pd.DataFrame([{")
    lines.append("    'var_smoothing': _var_smoothing,")
    lines.append("    'n_features': len(feat_cols),")
    lines.append("    'n_numeric': len(num_cols),")
    lines.append("    'n_categorical_used': len(cat_cols),")
    lines.append("    'skip_missing': _skip_missing,")
    lines.append("    'classes': ','.join(map(str, classes)),")
    lines.append("}])")
    return lines

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],   # Port 1 = training table
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python code body for the Naive Bayes learner.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: The generated Python code lines.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_nb_settings(ndir)

    # Input port
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0] if pairs else ("UNKNOWN", "1")

    # Outputs: default to two ports (1=bundle, 2=summary)
    ports = [str(p or "1") for p in (out_ports or ["1", "2"])] or ["1", "2"]

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"df = context['{src_id}:{in_port}']  # training table")
    lines.extend(_emit_train_code(cfg))

    # Publish results
    want = {"1": "bundle", "2": "summary_df"}
    remaining = ["bundle", "summary_df"]
    for p in ports:
        val = want.get(p)
        if val is None and remaining:
            val = remaining.pop(0)
        elif val in remaining:
            remaining.remove(val)
        if val:
            lines.append(f"context['{node_id}:{p}'] = {val}")

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
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines if this module can handle the node; None otherwise.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
