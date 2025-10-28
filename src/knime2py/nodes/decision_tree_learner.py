#!/usr/bin/env python3

"""Decision Tree Learner module.

Overview
----------------------------
This module generates Python code for a Decision Tree Learner node, which trains a 
scikit-learn DecisionTreeClassifier based on parameters parsed from settings.xml. 
It emits a model bundle (estimator + metadata), a feature-importance table, and a 
summary.

Runtime Behavior
----------------------------
Inputs:
- Reads a single DataFrame from the context, identified by the incoming port.

Outputs:
- Writes to context keys, mapping:
  - '1' → model bundle (dict containing estimator and metadata)
  - '2' → feature importances (DataFrame)
  - '3' → summary (DataFrame)

Key algorithms or mappings:
- Maps KNIME's split quality measures to sklearn's criterion options.
- Selects numeric and boolean columns for feature selection, excluding the target.

Edge Cases
----------------------------
The code handles cases such as missing target columns, empty DataFrames, and 
constant columns by providing fallback paths and default values.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries: pandas, numpy, 
scikit-learn. These dependencies are required for the generated code, not for 
this module.

Usage
----------------------------
This module is typically invoked by the knime2py emitter, which generates 
Python code for KNIME nodes. An example of expected context access is:
```python
df = context['source_id:1']  # input table
```

Node Identity
----------------------------
KNIME factory id: 
- FACTORY = "org.knime.base.node.mine.decisiontree2.learner2.DecisionTreeLearnerNodeFactory3"

Configuration
----------------------------
The settings are encapsulated in the `DecisionTreeSettings` dataclass, which 
includes important fields such as:
- target: The target column for classification (default: None).
- criterion: The criterion for splitting (default: "gini").
- min_samples_split: Minimum number of samples required to split an internal node (default: 2).
- random_state: Random seed for reproducibility (default: None).

The `parse_dt_settings` function extracts these values from settings.xml using 
XPath queries and provides fallbacks where necessary.

Limitations
----------------------------
Certain KNIME features, such as pruning methods and specific split strategies, 
are not supported or approximated in the sklearn implementation.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.node.mine.decisiontree2.learner2.DecisionTreeLearnerNodeFactory3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, normalize_in_ports, collect_module_imports, split_out_imports, iter_entries

# KNIME factory for Decision Tree Learner
FACTORY = "org.knime.base.node.mine.decisiontree2.learner2.DecisionTreeLearnerNodeFactory3"

# ---------------------------------------------------------------------
# settings.xml → DecisionTreeSettings
# ---------------------------------------------------------------------

@dataclass
class DecisionTreeSettings:
    target: Optional[str] = None
    criterion: str = "gini"          # sklearn: 'gini' | 'entropy' | 'log_loss'
    min_samples_split: int = 2       # KNIME "minNumberRecordsPerNode"
    random_state: Optional[int] = None
    # Parsed but not directly supported by sklearn's single-tree API
    pruning_method: Optional[str] = None           # e.g., 'MDL'
    reduced_error_pruning: bool = False
    binary_nominal_split: bool = False
    max_nominal_values: Optional[int] = None
    first_split_column: Optional[str] = None
    use_first_split: bool = False
    missing_strategy: Optional[str] = None         # e.g., 'lastPrediction'
    no_true_child_strategy: Optional[str] = None   # e.g., 'returnNullPrediction'


# KNIME → sklearn mapping for split quality
_CRITERION_MAP = {
    # KNIME names seen in UIs:
    #   "Gini", "Gini index" → gini
    #   "Information gain" → entropy
    #   "Gain ratio" (C4.5) → approx by entropy (no direct gain-ratio in sklearn)
    "GINI": "gini",
    "GINI INDEX": "gini",
    "INFORMATION GAIN": "entropy",
    "GAIN RATIO": "entropy",
    # Fallbacks
    "ENTROPY": "entropy",
}


def _to_int(s: Optional[str], default: int) -> int:
    """Convert a string to an integer, returning a default value if conversion fails."""
    try:
        return int(s) if s is not None else default
    except Exception:
        return default


def _to_bool(s: Optional[str], default: bool = False) -> bool:
    """Convert a string to a boolean, returning a default value if conversion fails."""
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def parse_dt_settings(node_dir: Optional[Path]) -> DecisionTreeSettings:
    """Parse the settings.xml file and return a DecisionTreeSettings object."""
    if not node_dir:
        return DecisionTreeSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return DecisionTreeSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return DecisionTreeSettings()

    target = first(model_el, ".//*[local-name()='entry' and @key='classifyColumn']/@value")

    # Split quality → sklearn criterion
    sq_raw = (first(model_el, ".//*[local-name()='entry' and @key='splitQualityMeasure']/@value") or "").strip().upper()
    criterion = _CRITERION_MAP.get(sq_raw, "gini")

    # Min records per node → min_samples_split
    min_split = _to_int(first(model_el, ".//*[local-name()='entry' and @key='minNumberRecordsPerNode']/@value"), 2)

    # Pruning & other flags (not directly supported; kept as informational)
    pruning_method = first(model_el, ".//*[local-name()='entry' and @key='pruningMethod']/@value")
    reduced_error = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='enableReducedErrorPruning']/@value"), False)
    binary_nominal = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='binaryNominalSplit']/@value"), False)
    max_nominal = first(model_el, ".//*[local-name()='entry' and @key='maxNumNominalValues']/@value")
    max_nominal = int(max_nominal) if (max_nominal and max_nominal.isdigit()) else None

    use_first_split = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='useFirstSplitColumn']/@value"), False)
    first_split_col = first(model_el, ".//*[local-name()='entry' and @key='firstSplitColumn']/@value")

    missing_strategy = first(model_el, ".//*[local-name()='entry' and @key='CFG_MISSINGSTRATEGY']/@value")
    no_true_child = first(model_el, ".//*[local-name()='entry' and @key='CFG_NOTRUECHILD']/@value")

    # KNIME node settings doesn't have random seed. Default is 1.
    seed = 1

    return DecisionTreeSettings(
        target=target or None,
        criterion=criterion,
        min_samples_split=max(2, min_split),
        random_state=seed,
        pruning_method=pruning_method or None,
        reduced_error_pruning=reduced_error,
        binary_nominal_split=binary_nominal,
        max_nominal_values=max_nominal,
        first_split_column=first_split_col or None,
        use_first_split=use_first_split,
        missing_strategy=missing_strategy or None,
        no_true_child_strategy=no_true_child or None,
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    """Generate a list of import statements required for the Decision Tree Learner."""
    return [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.tree import DecisionTreeClassifier",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.decisiontree2.learner2.DecisionTreeLearnerNodeFactory3"
)


def _emit_train_code(cfg: DecisionTreeSettings) -> List[str]:
    """
    Emit lines that:
      - select X, y (numeric/bool features by default, excluding target)
      - fit sklearn DecisionTreeClassifier
      - produce feature_importances table & a tiny summary
      - build a *bundle* dict with estimator + metadata for downstream Predictor
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    if not cfg.target:
        lines.append("# No target column configured; passthrough / no model.")
        lines.append("model = None")
        lines.append("importances_df = pd.DataFrame(columns=['feature','importance'])")
        lines.append("summary_df = pd.DataFrame([{'error': 'no target configured'}])")
        lines.append("bundle = {'estimator': None, 'features': [], 'feature_cols': [], 'target': None, 'classes': []}")
        return lines

    # Target column
    lines.append(f"_target = {repr(cfg.target)}")

    # Feature selection: numeric/boolean, excluding target
    lines.append("num_like = df.select_dtypes(include=['number','bool','boolean','Int64','Float64']).columns.tolist()")
    lines.append("feat_cols = [c for c in num_like if c != _target]")

    # Extract X, y
    lines.append("X = df[feat_cols].copy()")
    lines.append("y = df[_target].copy()")

    # Hyperparameters (map KNIME → sklearn)
    lines.append(f"_criterion = {repr(cfg.criterion)}  # mapped from KNIME splitQualityMeasure")
    lines.append(f"_min_samples_split = int({int(cfg.min_samples_split)})")
    lines.append("# KNIME Decision Tree Learner does not expose a random seed; k2p defaults to 1 for reproducibility.")
    lines.append(f"_seed = int({int(cfg.random_state)})  # default from k2p (1 unless overridden)")

    # Informational notes for unsupported knobs
    if cfg.pruning_method:
        lines.append(f"# KNIME pruningMethod={repr(cfg.pruning_method)} (not directly supported by sklearn DecisionTree; consider ccp_alpha)")
    if cfg.reduced_error_pruning:
        lines.append("# KNIME enableReducedErrorPruning=True (not supported in sklearn)")
    if cfg.use_first_split and cfg.first_split_column:
        lines.append(f"# KNIME useFirstSplitColumn=True, firstSplitColumn={repr(cfg.first_split_column)} (not supported in sklearn)")
    if cfg.binary_nominal_split:
        lines.append("# KNIME binaryNominalSplit=True (not applicable in sklearn)")
    if cfg.max_nominal_values is not None:
        lines.append(f"# KNIME maxNumNominalValues={int(cfg.max_nominal_values)} (not applicable in sklearn)")
    if cfg.missing_strategy:
        lines.append(f"# KNIME CFG_MISSINGSTRATEGY={repr(cfg.missing_strategy)} (sklearn requires pre-imputation)")
    if cfg.no_true_child_strategy:
        lines.append(f"# KNIME CFG_NOTRUECHILD={repr(cfg.no_true_child_strategy)}")

    # Fit model
    lines.append("model = DecisionTreeClassifier(")
    lines.append("    criterion=_criterion,")
    lines.append("    min_samples_split=_min_samples_split,")
    lines.append("    random_state=_seed,")
    lines.append(")")
    lines.append("model.fit(X, y)")

    # Feature importances
    lines.append("fi = getattr(model, 'feature_importances_', None)")
    lines.append("if fi is not None:")
    lines.append("    importances_df = pd.DataFrame({'feature': feat_cols, 'importance': fi})")
    lines.append("    importances_df = importances_df.sort_values('importance', ascending=False, kind='mergesort').reset_index(drop=True)")
    lines.append("else:")
    lines.append("    importances_df = pd.DataFrame(columns=['feature','importance'])")

    # Tiny summary
    lines.append(
        "summary_df = pd.DataFrame([{'criterion': _criterion, 'min_samples_split': _min_samples_split, "
        "'n_features': len(feat_cols), 'classes': ','.join(map(str, getattr(model, 'classes_', [])))}])"
    )

    # Bundle for downstream Predictor
    lines.append("bundle = {")
    lines.append("    'estimator': model,")
    lines.append("    'model': model,  # alias")
    lines.append("    'features': list(feat_cols),")
    lines.append("    'feature_cols': list(feat_cols),")
    lines.append("    'target': _target,")
    lines.append("    'classes': list(getattr(model, 'classes_', [])),")
    lines.append("    'criterion': _criterion,")
    lines.append("    'min_samples_split': _min_samples_split,")
    lines.append("    'random_state': _seed,")
    lines.append("}")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python code body for the Decision Tree Learner node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: The generated lines of Python code.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_dt_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Input (single table)
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    # Training + outputs
    lines.extend(_emit_train_code(cfg))

    # Publish:
    # Convention: 1=model bundle, 2=feature importances, 3=summary
    ports = [str(p or "1") for p in (out_ports or ["1", "2", "3"])]
    if not ports:
        ports = ["1", "2", "3"]

    want = {"1": "bundle", "2": "importances_df", "3": "summary_df"}
    remaining_vals = ["bundle", "importances_df", "summary_df"]
    for p in ports:
        val = want.get(p)
        if val is None and remaining_vals:
            val = remaining_vals.pop(0)
        elif val in remaining_vals:
            remaining_vals.remove(val)
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
    Generate the code for a Jupyter notebook cell for the Decision Tree Learner node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        str: The generated Jupyter notebook cell code.
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

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2", "3"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
