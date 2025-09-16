#!/usr/bin/env python3

####################################################################################################
#
# Logistic Regression Learner
#
# Trains a scikit-learn LogisticRegression from KNIME settings.xml, selecting X/y and publishing
# three outputs: (1) a model bundle for downstream prediction, (2) a coefficients table (with
# odds ratios), and (3) a compact training summary. Reads the first input table from context.
#
# - Feature selection: use included_names if set; otherwise all numeric/boolean columns minus the
#   target; then remove excluded_names.
# - Hyperparameter mapping: solver(KNIME→sklearn), maxEpoch→max_iter, epsilon→tol, seed→random_state.
#   Target reference category is recorded as metadata only (no sklearn equivalent).
#
####################################################################################################


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory for Logistic Regression Learner (v4)
FACTORY = "org.knime.base.node.mine.regression.logistic.learner4.LogRegLearnerNodeFactory4"

# ---------------------------------------------------------------------
# settings.xml → LogisticRegressionSettings
# ---------------------------------------------------------------------

@dataclass
class LogisticRegressionSettings:
    target: Optional[str] = None
    include_cols: List[str] = field(default_factory=list)
    exclude_cols: List[str] = field(default_factory=list)
    solver: str = "lbfgs"
    max_iter: int = 100
    tol: float = 1e-4
    seed: Optional[int] = None
    ref_category: Optional[str] = None  # KNIME "targetReferenceCategory" (informational)


def _collect_numeric_name_entries(cfg: ET._Element) -> List[str]:
    out: List[str] = []
    for k, v in iter_entries(cfg):
        if k.isdigit() and v:
            out.append(v.strip())
    return out


_SOLVER_MAP = {
    # KNIME → sklearn
    "SAG": "sag",
    "SAGA": "saga",
    "LBFGS": "lbfgs",
    "NEWTON-CG": "newton-cg",
    "LIBLINEAR": "liblinear",
}


def _to_int(s: Optional[str], default: int) -> int:
    try:
        return int(s) if s is not None else default
    except Exception:
        return default


def _to_float(s: Optional[str], default: float) -> float:
    try:
        return float(s) if s is not None else default
    except Exception:
        return default


def parse_logreg_settings(node_dir: Optional[Path]) -> LogisticRegressionSettings:
    if not node_dir:
        return LogisticRegressionSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return LogisticRegressionSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return LogisticRegressionSettings()

    target = first(model_el, ".//*[local-name()='entry' and @key='target']/@value")

    # Column filter includes/excludes
    include_cols: List[str] = []
    exclude_cols: List[str] = []
    cf = first_el(model_el, ".//*[local-name()='config' and @key='column-filter']")
    if cf is not None:
        inc_cfg = first_el(cf, ".//*[local-name()='config' and @key='included_names']")
        exc_cfg = first_el(cf, ".//*[local-name()='config' and @key='excluded_names']")
        if inc_cfg is not None:
            include_cols.extend(_collect_numeric_name_entries(inc_cfg))
        if exc_cfg is not None:
            exclude_cols.extend(_collect_numeric_name_entries(exc_cfg))

    # Solver / epochs / epsilon / seed
    solver_raw = (first(model_el, ".//*[local-name()='entry' and @key='solver']/@value") or "").strip().upper()
    solver = _SOLVER_MAP.get(solver_raw, "lbfgs")
    max_iter = _to_int(first(model_el, ".//*[local-name()='entry' and @key='maxEpoch']/@value"), 100)
    tol = _to_float(first(model_el, ".//*[local-name()='entry' and @key='epsilon']/@value"), 1e-4)

    seed_raw = first(model_el, ".//*[local-name()='entry' and @key='seed']/@value")
    try:
        seed = int(seed_raw) if seed_raw is not None else None
    except Exception:
        seed = None

    # Target reference category (best-effort)
    ref_category = None
    ref_el = first_el(model_el, ".//*[local-name()='config' and @key='targetReferenceCategory']")
    if ref_el is not None:
        rc = first(ref_el, ".//*[local-name()='entry' and contains(@key,'Cell')]/@value")
        if rc:
            ref_category = rc

    return LogisticRegressionSettings(
        target=target or None,
        include_cols=list(dict.fromkeys(include_cols)),
        exclude_cols=list(dict.fromkeys(exclude_cols)),
        solver=solver or "lbfgs",
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        ref_category=ref_category,
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    return [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.linear_model import LogisticRegression",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.regression.logistic.learner4.LogRegLearnerNodeFactory4"
)


def _emit_train_code(cfg: LogisticRegressionSettings) -> List[str]:
    """
    Emit lines that:
      - select X, y (using includes if present; else numeric columns minus target)
      - fit sklearn LogisticRegression
      - produce coef table & a small summary table
      - build a *bundle* dict with estimator + metadata for downstream Predictor
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    if not cfg.target:
        lines.append("# No target column configured; passthrough / no model.")
        lines.append("model = None")
        lines.append("coef_df = pd.DataFrame(columns=['feature','coefficient','odds_ratio'])")
        lines.append("summary_df = pd.DataFrame([{'error': 'no target configured'}])")
        lines.append("bundle = {'estimator': None, 'features': [], 'feature_cols': [], 'target': None, 'classes': []}")
        return lines

    tcol = repr(cfg.target)
    lines.append(f"_target = {tcol}")

    # Feature selection
    if cfg.include_cols:
        inc = ", ".join(repr(c) for c in cfg.include_cols)
        lines.append(f"_include = [{inc}]")
        lines.append("feat_cols = [c for c in _include if c in df.columns]")
    else:
        # default: use numeric/boolean columns except target
        lines.append("num_like = df.select_dtypes(include=['number','bool','boolean','Int64','Float64']).columns.tolist()")
        lines.append("feat_cols = [c for c in num_like if c != _target]")

    if cfg.exclude_cols:
        exc = ", ".join(repr(c) for c in cfg.exclude_cols)
        lines.append(f"_exclude = [{exc}]")
        lines.append("feat_cols = [c for c in feat_cols if c not in set(_exclude)]")

    lines.append("X = df[feat_cols].copy()")
    lines.append("y = df[_target].copy()")

    # Model hyperparams
    lines.append(f"_solver = {repr(cfg.solver)}")
    lines.append(f"_max_iter = {int(cfg.max_iter)}")
    lines.append(f"_tol = {float(cfg.tol)}")
    lines.append(f"_seed = {repr(cfg.seed)}")
    if cfg.ref_category:
        lines.append(f"# KNIME reference category (informational): {repr(cfg.ref_category)}")

    # Fit
    lines.append("model = LogisticRegression(solver=_solver, penalty='l2', max_iter=_max_iter, tol=_tol, random_state=_seed)")
    lines.append("model.fit(X, y)")

    # Coefs (+ odds ratio) including intercept as a special row
    lines.append("coefs = pd.Series(model.coef_.ravel(), index=feat_cols, name='coefficient')")
    lines.append("coef_df = pd.DataFrame({'feature': coefs.index, 'coefficient': coefs.values, 'odds_ratio': np.exp(coefs.values)})")
    lines.append("inter = float(model.intercept_.ravel()[0]) if getattr(model, 'intercept_', None) is not None else 0.0")
    lines.append("coef_df.loc[len(coef_df)] = ['__intercept__', inter, float(np.exp(inter))]")

    # A tiny summary frame
    lines.append(
        "summary_df = pd.DataFrame([{'solver': _solver, 'max_iter': _max_iter, 'tol': _tol, "
        "'n_features': len(feat_cols), 'classes': ','.join(map(str, getattr(model, 'classes_', [])))}])"
    )

    # ---- Build model bundle for downstream Predictor ----
    lines.append("bundle = {")
    lines.append("    'estimator': model,")             # preferred key
    lines.append("    'model': model,")                  # compatibility alias
    lines.append("    'features': list(feat_cols),")
    lines.append("    'feature_cols': list(feat_cols),") # compatibility alias
    lines.append("    'target': _target,")
    lines.append("    'classes': list(getattr(model, 'classes_', [])),")
    lines.append("    'solver': _solver, 'max_iter': _max_iter, 'tol': _tol, 'random_state': _seed,")
    lines.append("    'intercept_': float(inter),")
    lines.append("    'coef_': coefs.values.tolist(),")
    lines.append("}")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_logreg_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Input
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    # Training + outputs
    lines.extend(_emit_train_code(cfg))

    # Publish:
    # KNIME ports: 1=model bundle, 2=coefficients table, 3=summary table
    ports = [str(p or "1") for p in (out_ports or ["1", "2", "3"])]
    if not ports:
        ports = ["1", "2", "3"]

    want = {"1": "bundle", "2": "coef_df", "3": "summary_df"}
    remaining_vals = ["bundle", "coef_df", "summary_df"]

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

    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2", "3"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
