#!/usr/bin/env python3

####################################################################################################
#
# RProp MLP Learner
#
# Trains a scikit-learn MLPClassifier as a stand-in for KNIME’s RProp MLP, reading topology and run
# settings from settings.xml, selecting numeric/boolean features plus the configured target. Publishes
# (1) a model bundle and (2) a concise training summary.
#
# - Mapping: classcol→target; hiddenlayer→#hidden layers; nrhiddenneurons→neurons per layer;
#   maxiter→max_iter; ignoremv→drop rows with NA in X/y; useRandomSeed/randomSeed→random_state.
# - Topology: hidden_layer_sizes = [n_hidden_neurons] × n_hidden_layers.
# - Implementation detail: scikit-learn has no RProp; uses MLPClassifier (solver='adam') as an
#   approximation.
# - Features: all numeric/bool columns except target. If ignoremv=False, upstream imputation may be
#   required (sklearn MLP does not accept NaNs).
#
####################################################################################################


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, normalize_in_ports, collect_module_imports, split_out_imports, iter_entries

# KNIME factory for RProp MLP Learner (ANN)
FACTORY = "org.knime.base.node.mine.neural.rprop.RPropNodeFactory2"

# ---------------------------------------------------------------------
# settings.xml → RPropMLPSettings
# ---------------------------------------------------------------------

@dataclass
class RPropMLPSettings:
    target: Optional[str] = None
    n_hidden_layers: int = 1
    n_hidden_neurons: int = 16
    max_iter: int = 100
    ignore_missing: bool = False
    use_seed: bool = True
    random_state: int = 1


def _to_int(s: Optional[str], default: int) -> int:
    try:
        return int(s) if s is not None else default
    except Exception:
        return default


def _to_bool(s: Optional[str], default: bool = False) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def parse_rprop_settings(node_dir: Optional[Path]) -> RPropMLPSettings:
    if not node_dir:
        return RPropMLPSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return RPropMLPSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return RPropMLPSettings()

    target = first(model_el, ".//*[local-name()='entry' and @key='classcol']/@value") or None
    max_iter = _to_int(first(model_el, ".//*[local-name()='entry' and @key='maxiter']/@value"), 100)
    n_layers = _to_int(first(model_el, ".//*[local-name()='entry' and @key='hiddenlayer']/@value"), 1)
    n_neurons = _to_int(first(model_el, ".//*[local-name()='entry' and @key='nrhiddenneurons']/@value"), 16)
    ignoremv = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='ignoremv']/@value"), False)
    use_seed = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='useRandomSeed']/@value"), True)
    seed_val = _to_int(first(model_el, ".//*[local-name()='entry' and @key='randomSeed']/@value"), 1)

    return RPropMLPSettings(
        target=target,
        n_hidden_layers=max(1, n_layers),
        n_hidden_neurons=max(1, n_neurons),
        max_iter=max(1, max_iter),
        ignore_missing=ignoremv,
        use_seed=use_seed,
        random_state=seed_val if use_seed else 1,
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    return [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.neural_network import MLPClassifier",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.neural.rprop.RPropNodeFactory2"
)


def _emit_train_code(cfg: RPropMLPSettings) -> List[str]:
    """
    Emit lines that:
      - select X, y (numeric/bool features by default, excluding target)
      - fit sklearn MLPClassifier with topology mapped from KNIME
      - produce a small training summary
      - build a bundle dict with estimator + metadata for downstream prediction
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    if not cfg.target:
        lines.append("# No target column configured; passthrough / no model.")
        lines.append("model = None")
        lines.append("summary_df = pd.DataFrame([{'error': 'no target configured'}])")
        lines.append("bundle = {'estimator': None, 'features': [], 'feature_cols': [], 'target': None, 'classes': []}")
        return lines

    # Target & features
    lines.append(f"_target = {repr(cfg.target)}")
    lines.append("num_like = df.select_dtypes(include=['number','bool','boolean','Int64','Float64']).columns.tolist()")
    lines.append("feat_cols = [c for c in num_like if c != _target]")
    lines.append("X = df[feat_cols].copy()")
    lines.append("y = df[_target].copy()")

    # Missing values
    if cfg.ignore_missing:
        lines.append("# Drop rows with missing values in features/target (ignoremv=True)")
        lines.append("mask = X.notna().all(axis=1) & y.notna()")
        lines.append("X = X[mask]")
        lines.append("y = y[mask]")
        lines.append("out_df = out_df.loc[X.index]")
    else:
        lines.append("# Note: sklearn MLPClassifier does not accept NaNs; upstream imputation may be required.")

    # Topology & params
    lines.append(f"_hidden_layer_sizes = tuple([{cfg.n_hidden_neurons}] * {cfg.n_hidden_layers})")
    lines.append(f"_max_iter = int({cfg.max_iter})")
    lines.append(f"_seed = int({cfg.random_state})")

    # Train
    lines.append("model = MLPClassifier(")
    lines.append("    hidden_layer_sizes=_hidden_layer_sizes,")
    lines.append("    max_iter=_max_iter,")
    lines.append("    random_state=_seed,")
    lines.append("    # Note: sklearn has no RProp; using default solver='adam' as approximation")
    lines.append(")")
    lines.append("model.fit(X, y)")

    # Summary (safe attributes)
    lines.append("classes = list(getattr(model, 'classes_', []))")
    lines.append("n_iter_ = int(getattr(model, 'n_iter_', 0))")
    lines.append("loss_ = float(getattr(model, 'loss_', float('nan')))")
    lines.append("converged = bool(getattr(model, 'n_iter_', 0) < _max_iter)")
    lines.append("summary_df = pd.DataFrame([{" 
                 "'n_features': len(feat_cols), "
                 "'classes': ','.join(map(str, classes)), "
                 "'hidden_layer_sizes': str(_hidden_layer_sizes), "
                 "'max_iter': _max_iter, 'n_iter_': n_iter_, 'loss_': loss_, "
                 "'converged': converged"
                 "}])")

    # Bundle for downstream Predictor
    lines.append("bundle = {")
    lines.append("    'estimator': model,")
    lines.append("    'model': model,  # alias")
    lines.append("    'features': list(feat_cols),")
    lines.append("    'feature_cols': list(feat_cols),")
    lines.append("    'target': _target,")
    lines.append("    'classes': classes,")
    lines.append("    'hidden_layer_sizes': _hidden_layer_sizes,")
    lines.append("    'max_iter': _max_iter,")
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
    cfg = parse_rprop_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Input (single table)
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    # Training + outputs
    lines.extend(_emit_train_code(cfg))

    # Publish:
    # Convention: 1=model bundle, 2=summary
    ports = [str(p or "1") for p in (out_ports or ["1", "2"])]
    if not ports:
        ports = ["1", "2"]

    want = {"1": "bundle", "2": "summary_df"}
    remaining_vals = ["bundle", "summary_df"]
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
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
