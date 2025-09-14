#!/usr/bin/env python3

####################################################################################################
#
# Gradient Boosted Trees (Classification) Learner
#
# Trains a scikit-learn GradientBoostingClassifier from KNIME settings.xml, selecting features and
# target, then publishes three outputs: (1) a model bundle for downstream prediction, (2) a
# feature-importance table, and (3) a compact training summary. Inputs are read from the first
# table port; results are written into the node's context ports.
#
# - Feature selection: use included_names if present; otherwise all numeric/boolean columns except
#   the target. Excluded_names are removed afterward. If no target is configured, the node is a
#   passthrough: bundle=None and empty outputs with an error note in the summary.
# - Hyperparameters mapped: nrModels→n_estimators, learningRate→learning_rate, maxLevels
#   (-1/absent → default 3)→max_depth, minNodeSize→min_samples_split (≥2), minChildSize→min_samples_leaf (≥1),
#   dataFraction (0<≤1)→subsample (stochastic GB), columnSamplingMode→max_features (None/'sqrt'/'log2'/fraction/int),
#   seed→random_state. Seed defaults to 1 for deterministic output.
# - Unsupported/orthogonal flags: splitCriterion (trees in sklearn GBT have fixed criterion),
#   missingValueHandling (impute beforehand), useAverageSplitPoints, useBinaryNominalSplits,
#   isUseDifferentAttributesAtEachNode (no direct sklearn analog). These are noted and ignored.
# - Outputs: port 1=model bundle (estimator, metadata), port 2=feature_importances_, port 3=summary.
# - Dependencies: lxml for XML parsing; pandas/numpy for data handling; scikit-learn for modeling.
#
####################################################################################################


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory for Gradient Boosted Trees (classification) Learner (Tree Ensembles v2)
GBT_FACTORY = "org.knime.base.node.mine.treeensemble2.node.gradientboosting.learner.classification.GradientBoostingClassificationLearnerNodeFactory2"


def can_handle(node_type: Optional[str]) -> bool:
    return bool(node_type and node_type.endswith(GBT_FACTORY))


# ---------------------------------------------------------------------
# settings.xml → GradientBoostingSettings
# ---------------------------------------------------------------------

@dataclass
class GradientBoostingSettings:
    target: Optional[str] = None
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: Optional[int] = 3
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    subsample: float = 1.0                 # KNIME dataFraction (0<subsample<=1 => stochastic GB)
    max_features: Optional[object] = None  # mapped from columnSamplingMode
    random_state: int = 1                  # KNIME seed (fallback to 1)

    # Info-only (not directly mapped in sklearn GBT)
    split_criterion_raw: Optional[str] = None
    missing_value_handling: Optional[str] = None
    use_average_split_points: bool = False
    use_binary_nominal_splits: bool = False
    use_diff_attrs_each_node: bool = False

    include_cols: List[str] = None
    exclude_cols: List[str] = None


def _to_int(s: Optional[str], default: int) -> int:
    try:
        return int(s) if s is not None else default
    except Exception:
        return default


def _to_float(s: Optional[str], default: Optional[float]) -> Optional[float]:
    try:
        return float(s) if s is not None else default
    except Exception:
        return default


def _to_bool(s: Optional[str], default: bool = False) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def _collect_name_entries(cfg: ET._Element) -> List[str]:
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
    KNIME 'columnSamplingMode' → sklearn GradientBoostingClassifier(max_features)

    - None/All      → None (use all features)
    - SquareRoot    → 'sqrt'
    - Log2          → 'log2'
    - Fraction      → float in (0,1]
    - Absolute      → int
    Note: KNIME's 'isUseDifferentAttributesAtEachNode' has no sklearn analog here.
    """
    m = (mode or "").strip().upper()
    if m in {"NONE", "ALL", "ALLCOLUMNS", "ALL_COLUMNS", ""}:
        return None
    if m in {"SQUAREROOT", "SQRT"}:
        return "sqrt"
    if m == "LOG2":
        return "log2"
    if m == "FRACTION":
        if frac is not None and 0.0 < frac <= 1.0:
            return float(frac)
        return None
    if m == "ABSOLUTE":
        if isinstance(absolute, int) and absolute > 0:
            return int(absolute)
        return None
    return None


def parse_gbt_settings(node_dir: Optional[Path]) -> GradientBoostingSettings:
    if not node_dir:
        return GradientBoostingSettings(include_cols=[], exclude_cols=[])

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return GradientBoostingSettings(include_cols=[], exclude_cols=[])

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return GradientBoostingSettings(include_cols=[], exclude_cols=[])

    target = first(model_el, ".//*[local-name()='entry' and @key='targetColumn']/@value")

    seed = _to_int(first(model_el, ".//*[local-name()='entry' and @key='seed']/@value"), 1)
    n_estimators = _to_int(first(model_el, ".//*[local-name()='entry' and @key='nrModels']/@value"), 100)
    learning_rate = _to_float(first(model_el, ".//*[local-name()='entry' and @key='learningRate']/@value"), 0.1) or 0.1

    # Depth / min samples (KNIME -1 means "use default")
    max_levels = first(model_el, ".//*[local-name()='entry' and @key='maxLevels']/@value")
    max_depth = None
    if max_levels and max_levels.lstrip("-").isdigit():
        v = int(max_levels)
        max_depth = v if v > 0 else 3  # sklearn default for GBT is 3

    min_node_size = _to_int(first(model_el, ".//*[local-name()='entry' and @key='minNodeSize']/@value"), -1)
    min_child_size = _to_int(first(model_el, ".//*[local-name()='entry' and @key='minChildSize']/@value"), -1)
    min_samples_split = max(2, min_node_size) if min_node_size and min_node_size > 0 else 2
    min_samples_leaf = max(1, min_child_size) if min_child_size and min_child_size > 0 else 1

    # Row sampling → subsample (no replacement in sklearn GBT)
    data_fraction = _to_float(first(model_el, ".//*[local-name()='entry' and @key='dataFraction']/@value"), 1.0) or 1.0
    subsample = data_fraction if 0.0 < data_fraction <= 1.0 else 1.0

    # Column sampling → max_features
    col_mode = first(model_el, ".//*[local-name()='entry' and @key='columnSamplingMode']/@value") or "None"
    col_frac = _to_float(first(model_el, ".//*[local-name()='entry' and @key='columnFractionPerTree']/@value"), None)
    col_abs  = _to_int(first(model_el, ".//*[local-name()='entry' and @key='columnAbsolutePerTree']/@value"), None)
    use_diff_attrs = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='isUseDifferentAttributesAtEachNode']/@value"), False)
    max_features = _map_max_features(col_mode, col_frac, col_abs, use_diff_attrs)

    # Informational flags
    split_criterion_raw = first(model_el, ".//*[local-name()='entry' and @key='splitCriterion']/@value")
    missing_value_handling = first(model_el, ".//*[local-name()='entry' and @key='missingValueHandling']/@value")
    use_avg_split = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='useAverageSplitPoints']/@value"), False)
    use_bin_nom = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='useBinaryNominalSplits']/@value"), False)

    # Column filter includes/excludes
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

    return GradientBoostingSettings(
        target=target or None,
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=max_depth,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        subsample=float(subsample),
        max_features=max_features,
        random_state=int(seed),
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
    return [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.ensemble import GradientBoostingClassifier",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.ensembles/latest/"
    "org.knime.base.node.mine.treeensemble2.node.gradientboosting.learner.classification.GradientBoostingClassificationLearnerNodeFactory2"
)


def _emit_train_code(cfg: GradientBoostingSettings) -> List[str]:
    """
    Emit code that:
      - selects X, y (use included_names if provided; else numeric/bool excluding target)
      - fits sklearn GradientBoostingClassifier
      - builds feature_importances and a small summary
      - bundles estimator + metadata for downstream Predictor
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
    lines.append(f"_learning_rate = float({float(cfg.learning_rate)})")
    lines.append(f"_max_depth = {repr(cfg.max_depth if cfg.max_depth is not None else 3)}")
    lines.append(f"_min_samples_split = int({int(cfg.min_samples_split)})")
    lines.append(f"_min_samples_leaf = int({int(cfg.min_samples_leaf)})")
    lines.append(f"_subsample = float({float(cfg.subsample)})")
    lines.append(f"_max_features = {repr(cfg.max_features)}  # mapped from KNIME columnSamplingMode (None => all features)")
    lines.append(f"_seed = int({int(cfg.random_state)})")

    # Info-only notes (not directly supported / orthogonal)
    if cfg.split_criterion_raw:
        lines.append(f"# KNIME splitCriterion={repr(cfg.split_criterion_raw)} (sklearn GBT uses tree regressors internally; criterion is fixed)")
    if cfg.missing_value_handling:
        lines.append(f"# KNIME missingValueHandling={repr(cfg.missing_value_handling)} (not supported in sklearn GBT; impute beforehand)")
    if cfg.use_average_split_points:
        lines.append("# KNIME useAverageSplitPoints=True (not applicable in sklearn GBT)")
    if cfg.use_binary_nominal_splits:
        lines.append("# KNIME useBinaryNominalSplits=True (not applicable in sklearn GBT)")
    if cfg.use_diff_attrs_each_node:
        lines.append("# KNIME isUseDifferentAttributesAtEachNode=True (no direct equivalent in sklearn; ignored)")

    # Fit model
    lines.append("model = GradientBoostingClassifier(")
    lines.append("    n_estimators=_n_estimators,")
    lines.append("    learning_rate=_learning_rate,")
    lines.append("    max_depth=_max_depth,")
    lines.append("    min_samples_split=_min_samples_split,")
    lines.append("    min_samples_leaf=_min_samples_leaf,")
    lines.append("    subsample=_subsample,")
    lines.append("    max_features=_max_features,")
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

    # Summary
    lines.append(
        "summary_df = pd.DataFrame([{'n_estimators': _n_estimators, 'learning_rate': _learning_rate, "
        "'max_depth': _max_depth, 'min_samples_split': _min_samples_split, 'min_samples_leaf': _min_samples_leaf, "
        "'subsample': _subsample, 'max_features': _max_features, "
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
    lines.append("    'learning_rate': _learning_rate,")
    lines.append("    'max_depth': _max_depth,")
    lines.append("    'min_samples_split': _min_samples_split,")
    lines.append("    'min_samples_leaf': _min_samples_leaf,")
    lines.append("    'subsample': _subsample,")
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
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_gbt_settings(ndir)

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
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Returns (imports, body_lines) if this module can handle the node; None otherwise.
    """
    if not (ntype and can_handle(ntype)):
        return None

    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2", "3"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
