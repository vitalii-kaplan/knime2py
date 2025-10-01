#!/usr/bin/env python3

####################################################################################################
#
# RProp MLP Learner (improved approximation of KNIME’s RProp)
#
# Reads topology and run settings from settings.xml and trains an MLPClassifier. To better mirror
# KNIME’s behavior WITHOUT changing any settings-derived values:
#   • Continuous features: median impute + StandardScaler (z-score).
#   • Binary/one-hot features: most_frequent impute (no scaling).
#   • Activation='tanh', solver='lbfgs' (scikit-learn has no RProp; this is a closer stand-in).
#
# Mapping (unchanged):
#   - classcol → target
#   - hiddenlayer → #hidden layers
#   - nrhiddenneurons → neurons per layer
#   - maxiter → max_iter
#   - ignoremv → if True, drop rows with NA in X/y; if False, use imputers (no setting change)
#   - useRandomSeed/randomSeed → random_state
#
# Outputs:
#   - Port 1: model bundle dict {'estimator','features','target','classes',...}
#   - Port 2: summary_df with basic training metadata
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, normalize_in_ports, collect_module_imports, split_out_imports, iter_entries

FACTORY = "org.knime.base.node.mine.neural.rprop.RPropNodeFactory2"

# --------------------------------------------------------------------------------------------------
# settings.xml → RPropMLPSettings
# --------------------------------------------------------------------------------------------------

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

    target    = first(model_el, ".//*[local-name()='entry' and @key='classcol']/@value") or None
    max_iter  = _to_int(first(model_el, ".//*[local-name()='entry' and @key='maxiter']/@value"), 100)
    n_layers  = _to_int(first(model_el, ".//*[local-name()='entry' and @key='hiddenlayer']/@value"), 1)
    n_neurons = _to_int(first(model_el, ".//*[local-name()='entry' and @key='nrhiddenneurons']/@value"), 16)
    ignoremv  = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='ignoremv']/@value"), False)
    use_seed  = _to_bool(first(model_el, ".//*[local-name()='entry' and @key='useRandomSeed']/@value"), True)
    seed_val  = _to_int(first(model_el, ".//*[local-name()='entry' and @key='randomSeed']/@value"), 1)

    return RPropMLPSettings(
        target=target,
        n_hidden_layers=max(1, n_layers),
        n_hidden_neurons=max(1, n_neurons),
        max_iter=max(1, max_iter),
        ignore_missing=ignoremv,
        use_seed=use_seed,
        random_state=seed_val if use_seed else 1,
    )

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    return [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.neural_network import MLPClassifier",
        "from sklearn.pipeline import Pipeline, make_pipeline",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.preprocessing import StandardScaler",
        "from sklearn.impute import SimpleImputer",
    ]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.neural.rprop.RPropNodeFactory2"
)

def _emit_train_code(cfg: RPropMLPSettings) -> List[str]:
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    if not cfg.target:
        lines += [
            "# No target column configured; passthrough / no model.",
            "model = None",
            "summary_df = pd.DataFrame([{'error': 'no target configured'}])",
            "bundle = {'estimator': None, 'features': [], 'feature_cols': [], 'target': None, 'classes': []}",
        ]
        return lines

    # Target & features
    lines.append(f"_target = {repr(cfg.target)}")
    lines.append("if _target not in df.columns:")
    lines.append("    raise KeyError(f'MLP Learner: target column not found: {_target!r}')")
    lines.append("num_like = df.select_dtypes(include=['number','bool','boolean','Int64','Float64']).columns.tolist()")
    lines.append("feat_cols = [c for c in num_like if c != _target]")
    lines.append("if not feat_cols:")
    lines.append("    raise ValueError('MLP Learner: no numeric/bool feature columns found (after excluding target)')")

    lines.append("X = df[feat_cols].copy()")
    lines.append("y = df[_target].copy()")

    # Missing values handling per setting
    if cfg.ignore_missing:
        lines.append("# ignoremv=True → drop rows with missing values in X or y")
        lines.append("mask = X.notna().all(axis=1) & y.notna()")
        lines.append("X = X[mask]")
        lines.append("y = y[mask]")
        lines.append("out_df = out_df.loc[X.index]")
        # If everything got dropped:
        lines.append("if X.shape[0] == 0:")
        lines.append("    raise ValueError('MLP Learner: no rows left after dropping missing values')")
    else:
        lines.append("# ignoremv=False → keep rows; impute inside the pipeline so sklearn can fit")

    # Identify binary vs continuous feature columns
    lines.append("def _is_binary_col(s):")
    lines.append("    try:")
    lines.append("        v = pd.to_numeric(s, errors='coerce')")
    lines.append("    except Exception:")
    lines.append("        return False")
    lines.append("    u = pd.unique(v.dropna())")
    lines.append("    return set(map(float, u)) <= {0.0, 1.0} and len(u) <= 2")
    lines.append("")
    lines.append("_bin_cols = [c for c in feat_cols if _is_binary_col(X[c])]")
    lines.append("_cont_cols = [c for c in feat_cols if c not in _bin_cols]")

    # Preprocessing pipelines
    lines.append("bin_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent'))])")
    lines.append("cont_pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])")
    lines.append("ct = ColumnTransformer([")
    lines.append("    ('bin',  bin_pipe, _bin_cols),")
    lines.append("    ('cont', cont_pipe, _cont_cols),")
    lines.append("], remainder='drop', sparse_threshold=0.0)")
    lines.append("")

    # Topology & params from settings
    lines.append(f"_hidden_layer_sizes = tuple([{cfg.n_hidden_neurons}] * {cfg.n_hidden_layers})")
    lines.append(f"_max_iter = int({cfg.max_iter})")
    lines.append(f"_seed = int({cfg.random_state})")

    # MLP: closer to RProp behavior (tanh + lbfgs)
    lines.append("mlp = MLPClassifier(")
    lines.append("    hidden_layer_sizes=_hidden_layer_sizes,")
    lines.append("    activation='tanh',")
    lines.append("    solver='lbfgs',")
    lines.append("    max_iter=_max_iter,")
    lines.append("    random_state=_seed,")
    lines.append(")")
    lines.append("model = make_pipeline(ct, mlp)")
    lines.append("model.fit(X, y)")

    # Summary (pull from final step)
    lines.append("classes = list(getattr(model[-1], 'classes_', []))")
    lines.append("n_iter_ = int(getattr(model[-1], 'n_iter_', 0))")
    lines.append("loss_ = float(getattr(model[-1], 'loss_', float('nan')))")
    lines.append("converged = bool(n_iter_ < _max_iter)  # lbfgs usually converges before max_iter")
    lines.append("summary_df = pd.DataFrame([{'n_features': len(feat_cols),")
    lines.append("                               'classes': ','.join(map(str, classes)),")
    lines.append("                               'hidden_layer_sizes': str(_hidden_layer_sizes),")
    lines.append("                               'max_iter': _max_iter, 'n_iter_': n_iter_,")
    lines.append("                               'loss_': loss_, 'converged': converged}])")

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

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_train_code(cfg))

    # Publish: 1=model bundle, 2=summary
    ports = [str(p or '1') for p in (out_ports or ['1', '2'])]
    if not ports:
        ports = ['1', '2']

    want = {'1': 'bundle', '2': 'summary_df'}
    remaining_vals = ['bundle', 'summary_df']
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
    explicit_imports = collect_module_imports(generate_imports)

    in_ports  = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
