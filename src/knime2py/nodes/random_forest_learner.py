#!/usr/bin/env python3

####################################################################################################
#
# Random Forest (Classification) Learner
#
# Trains a scikit-learn RandomForestClassifier from KNIME settings.xml, selecting features/target,
# then publishes three outputs: (3) a model bundle for downstream prediction, (1) a feature-
# importance table, and (2) a compact training summary. Reads the first input table from context.
#
# - Feature selection: use included_names if provided; otherwise all numeric/boolean columns except
#   the target; excluded_names are removed afterward.
# - Hyperparameter mapping: nrModels→n_estimators; maxLevels>0→max_depth else None; minNodeSize→min_samples_split;
#   minChildSize→min_samples_leaf; isDataSelectionWithReplacement→bootstrap; dataFraction→max_samples
#   (only when bootstrap=True); columnSamplingMode/columnFractionPerTree/columnAbsolutePerTree plus
#   isUseDifferentAttributesAtEachNode→max_features ('sqrt'/'log2'/1.0/fraction/int); seed→random_state.
# - Info-only flags (not applied in sklearn RF): splitCriterion, missingValueHandling,
#   useAverageSplitPoints, useBinaryNominalSplits; noted and ignored.
# 
####################################################################################################


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory for Random Forest Classification Learner (Tree Ensembles v2)
FACTORY = "org.knime.base.node.mine.treeensemble2.node.randomforest.learner.classification.RandomForestClassificationLearnerNodeFactory2"

# ---------------------------------------------------------------------
# settings.xml → RandomForestSettings
# ---------------------------------------------------------------------

@dataclass
class RandomForestSettings:
    target: Optional[str] = None
    n_estimators: int = 100
    max_depth: Optional[int] = None         # KNIME maxLevels (None => unlimited)
    min_samples_split: int = 2              # KNIME minNodeSize
    min_samples_leaf: int = 1               # KNIME minChildSize
    bootstrap: bool = True                  # KNIME isDataSelectionWithReplacement
    max_samples: Optional[float] = None     # KNIME dataFraction (used only if bootstrap=True)
    max_features: Optional[object] = "sqrt" # KNIME columnSamplingMode (+ optional *_PerTree)
    random_state: int = 1                   # KNIME seed (fallback to 1 if missing)
    # Informational (not directly supported in sklearn RF)
    split_criterion_raw: Optional[str] = None
    missing_value_handling: Optional[str] = None
    use_average_split_points: bool = False
    use_binary_nominal_splits: bool = False
    use_diff_attrs_each_node: bool = True
    include_cols: List[str] = field(default_factory=list)
    exclude_cols: List[str] = field(default_factory=list)


def _to_int(s: Optional[str], default: int) -> int:
    """Convert a string to an integer, returning a default value if conversion fails."""
    try:
        return int(s) if s is not None else default
    except Exception:
        return default


def _to_float(s: Optional[str], default: Optional[float]) -> Optional[float]:
    """Convert a string to a float, returning a default value if conversion fails."""
    try:
        return float(s) if s is not None else default
    except Exception:
        return default


def _to_bool(s: Optional[str], default: bool = False) -> bool:
    """Convert a string to a boolean, returning a default value if conversion fails."""
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def _collect_name_entries(cfg: ET._Element) -> List[str]:
    """Collect non-empty name entries from the given XML element."""
    out: List[str] = []
    for k, v in iter_entries(cfg):
        if k.isdigit() and v:
            out.append(v.strip())
    return out


def _map_max_features(mode: str,
                      frac: Optional[float],
                      absolute: Optional[int],
                      use_diff_attrs: bool) -> Optional[object]:
    """
    Map KNIME 'columnSamplingMode' to sklearn RandomForestClassifier(max_features).

    - SquareRoot  → 'sqrt'
    - Log2        → 'log2'
    - All         → 1.0 (all features)
    - Fraction    → float in (0,1]
    - Absolute    → int
    If 'use_diff_attrs' is False, KNIME chooses attributes once per tree,
    which sklearn doesn't support directly. We approximate by 'all features'.
    """
    mode_up = (mode or "").strip().upper()
    if not use_diff_attrs:
        return 1.0

    if mode_up in {"SQUAREROOT", "SQRT"}:
        return "sqrt"
    if mode_up == "LOG2":
        return "log2"
    if mode_up in {"ALL", "ALLCOLUMNS", "ALL_COLUMNS"}:
        return 1.0
    if mode_up == "FRACTION":
        if frac is not None and 0.0 < frac <= 1.0:
            return float(frac)
        return "sqrt"
    if mode_up == "ABSOLUTE":
        if isinstance(absolute, int) and absolute > 0:
            return int(absolute)
        return "sqrt"
    return "sqrt"


def parse_rf_settings(node_dir: Optional[Path]) -> RandomForestSettings:
    """Parse the Random Forest settings from the settings.xml file in the given directory."""
    if not node_dir:
        return RandomForestSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return RandomForestSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return RandomForestSettings()

    target = first(model_el, ".//*[local-name()='entry' and @key='targetColumn']/@value")

    seed = _to_int(first(model_el, ".//*[local-name()='entry' and @key='seed']/@value"), 1)
    n_estimators = _to_int(first(model_el, ".//*[local-name()='entry' and @key='nrModels']/@value"), 100)

    max_levels = first(model_el, ".//*[local-name()='entry' and @key='maxLevels']/@value")
    max_depth = int(max_levels) if (max_levels and max_levels.isdigit() and int(max_levels) > 0) else None

    min_node_size = _to_int(first(model_el, ".//*[local-name()='entry' and @key='minNodeSize']/@value"), 2)
    min_child_size = _to_int(first(model_el, ".//*[local-name()='entry' and @key='minChildSize']/@value"), 1)

    data_fraction = _to_float(first(model_el, ".//*[local-name()='entry' and @key='dataFraction']/@value"), 1.0)
    bootstrap = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='isDataSelectionWithReplacement']/@value"), True)

    col_mode = first(model_el, ".//*[local-name()='entry' and @key='columnSamplingMode']/@value") or "SquareRoot"
    col_frac = _to_float(first(model_el, ".//*[local-name()='entry' and @key='columnFractionPerTree']/@value"), None)
    col_abs  = _to_int(first(model_el, ".//*[local-name()='entry' and @key='columnAbsolutePerTree']/@value"), None)
    use_diff_attrs = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='isUseDifferentAttributesAtEachNode']/@value"), True)

    split_criterion_raw = first(model_el, ".//*[local-name()='entry' and @key='splitCriterion']/@value")
    missing_value_handling = first(model_el, ".//*[local-name()='entry' and @key='missingValueHandling']/@value")
    use_avg_split = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='useAverageSplitPoints']/@value"), False)
    use_bin_nom = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='useBinaryNominalSplits']/@value"), False)

    # Column filter includes/excludes (if present)
    include_cols: List[str] = []
    exclude_cols: List[str] = []
    cfc = first_el(model_el, ".//*[local-name()='config' and @key='columnFilterConfig']")
    if cfc is not None:
        inc_cfg = first_el(cfc, ".//*[local-name()='config' and @key='included_names']")
        exc_cfg = first_el(cfc, ".//*[local-name()='config' and @key='excluded_names']")
        if inc_cfg is not None:
            include_cols.extend(_collect_name_entries(inc_cfg))
        if exc_cfg is not None:
            exclude_cols.extend(_collect_name_entries(exc_cfg))
    include_cols = list(dict.fromkeys(include_cols))
    exclude_cols = list(dict.fromkeys(exclude_cols))

    # Map KNIME column sampling -> sklearn max_features
    max_features = _map_max_features(col_mode, col_frac, col_abs, use_diff_attrs)

    # max_samples only works if bootstrap=True (sklearn)
    max_samples = None
    if bootstrap and data_fraction is not None and 0.0 < data_fraction < 1.0:
        max_samples = float(data_fraction)

    return RandomForestSettings(
        target=target or None,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=max(2, min_node_size),
        min_samples_leaf=max(1, min_child_size),
        bootstrap=bootstrap,
        max_samples=max_samples,
        max_features=max_features,
        random_state=seed,
        split_criterion_raw=split_criterion_raw or None,
        missing_value_handling=missing_value_handling or None,
        use_average_split_points=use_avg_split,
        use_binary_nominal_splits=use_bin_nom,
        use_diff_attrs_each_node=use_diff_attrs,
        include_cols=include_cols,
        exclude_cols=exclude_cols,
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    """Generate import statements for the Random Forest learner."""
    return [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.ensemble import RandomForestClassifier",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.ensembles/latest/"
    "org.knime.base.node.mine.treeensemble2.node.randomforest.learner.classification.RandomForestClassificationLearnerNodeFactory2"
)


def _emit_train_code(cfg: RandomForestSettings) -> List[str]:
    """
    Emit lines that:
      - select X, y (use included names if provided; else numeric/bool)
      - fit sklearn RandomForestClassifier
      - produce feature_importances table & a small summary
      - build a *bundle* dict with estimator + metadata for downstream Predictor

    Args:
        cfg (RandomForestSettings): The configuration settings for the Random Forest model.

    Returns:
        List[str]: The lines of code to train the model and generate outputs.
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

    # Target
    lines.append(f"_target = {repr(cfg.target)}")

    # Feature selection
    if cfg.include_cols:
        inc = ", ".join(repr(c) for c in cfg.include_cols)
        lines.append(f"_include = [{inc}]")
        lines.append("feat_cols = [c for c in _include if c in df.columns]")
    else:
        lines.append("num_like = df.select_dtypes(include=['number','bool','boolean','Int64','Float64']).columns.tolist()")
        lines.append("feat_cols = [c for c in num_like if c != _target]")
    if cfg.exclude_cols:
        exc = ", ".join(repr(c) for c in cfg.exclude_cols)
        lines.append(f"_exclude = [{exc}]")
        lines.append("feat_cols = [c for c in feat_cols if c not in set(_exclude)]")

    lines.append("X = df[feat_cols].copy()")
    lines.append("y = df[_target].copy()")

    # Hyperparameters
    lines.append(f"_n_estimators = int({int(cfg.n_estimators)})")
    lines.append(f"_max_depth = {repr(cfg.max_depth)}")
    lines.append(f"_min_samples_split = int({int(cfg.min_samples_split)})")
    lines.append(f"_min_samples_leaf = int({int(cfg.min_samples_leaf)})")
    lines.append(f"_bootstrap = {repr(bool(cfg.bootstrap))}")
    # sklearn uses max_samples only if bootstrap=True; float in (0,1] means fraction of n_samples.
    lines.append(f"_max_samples = {repr(cfg.max_samples)}")
    lines.append(f"_max_features = {repr(cfg.max_features)}  # mapped from KNIME columnSamplingMode")
    lines.append(f"_seed = int({int(cfg.random_state)})  # from KNIME settings (defaulted to 1 if missing)")

    # Info-only notes
    if cfg.split_criterion_raw:
        lines.append(f"# KNIME splitCriterion={repr(cfg.split_criterion_raw)} (sklearn RF supports 'gini'/'entropy' internally)")
    if cfg.missing_value_handling:
        lines.append(f"# KNIME missingValueHandling={repr(cfg.missing_value_handling)} (not supported in sklearn RF; impute beforehand)")
    if cfg.use_average_split_points:
        lines.append("# KNIME useAverageSplitPoints=True (not applicable in sklearn RF)")
    if cfg.use_binary_nominal_splits:
        lines.append("# KNIME useBinaryNominalSplits=True (not applicable in sklearn RF)")
    if not cfg.use_diff_attrs_each_node:
        lines.append("# KNIME isUseDifferentAttributesAtEachNode=False (no direct equivalent in sklearn; approximated by max_features=1.0)")

    # Fit model
    lines.append("model = RandomForestClassifier(")
    lines.append("    n_estimators=_n_estimators,")
    lines.append("    max_depth=_max_depth,")
    lines.append("    min_samples_split=_min_samples_split,")
    lines.append("    min_samples_leaf=_min_samples_leaf,")
    lines.append("    bootstrap=_bootstrap,")
    lines.append("    max_samples=_max_samples,")
    lines.append("    max_features=_max_features,")
    lines.append("    random_state=_seed,")
    lines.append("    n_jobs=-1,")
    lines.append(")")
    lines.append("model.fit(X, y)")

    # Feature importances
    lines.append("fi = getattr(model, 'feature_importances_', None)")
    lines.append("if fi is not None:")
    lines.append("    importances_df = pd.DataFrame({'feature': feat_cols, 'importance': fi})")
    lines.append("    importances_df = importances_df.sort_values('importance', ascending=False, kind='mergesort').reset_index(drop=True)")
    lines.append("else:")
    lines.append("    importances_df = pd.DataFrame(columns=['feature','importance'])")

    # Summary
    lines.append(
        "summary_df = pd.DataFrame([{'n_estimators': _n_estimators, 'max_depth': _max_depth, "
        "'min_samples_split': _min_samples_split, 'min_samples_leaf': _min_samples_leaf, "
        "'bootstrap': _bootstrap, 'max_samples': _max_samples, 'max_features': _max_features, "
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
    lines.append("    'n_estimators': _n_estimators,")
    lines.append("    'max_depth': _max_depth,")
    lines.append("    'min_samples_split': _min_samples_split,")
    lines.append("    'min_samples_leaf': _min_samples_leaf,")
    lines.append("    'bootstrap': _bootstrap,")
    lines.append("    'max_samples': _max_samples,")
    lines.append("    'max_features': _max_features,")
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
    Generate the Python code body for the Random Forest learner.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: The lines of code for the node's functionality.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_rf_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Input (single table)
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    # Training + outputs
    lines.extend(_emit_train_code(cfg))

    # Publish:
    # NOTE (TEv2): the **model** is on KNIME port 3.
    # We'll map outputs accordingly:
    #   - port 3 → bundle (model)
    #   - port 1 → feature importances
    #   - port 2 → summary
    ports = [str(p or "1") for p in (out_ports or ["1", "2", "3"])]
    if not ports:
        ports = ["1", "2", "3"]

    want = {"3": "bundle", "1": "importances_df", "2": "summary_df"}
    remaining = ["bundle", "importances_df", "summary_df"]
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
    Generate the code for a Jupyter notebook cell for the Random Forest learner.

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
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines if this module can handle the node; None otherwise.
    """

    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2", "3"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
