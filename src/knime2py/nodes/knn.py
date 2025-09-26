#!/usr/bin/env python3

####################################################################################################
#
# K Nearest Neighbor (single-node trainer + scorer)
#
# Trains a scikit-learn KNeighborsClassifier from KNIME settings.xml and scores the same input
# table. Appends:
#   • "Class [kNN]"                          ← KNIME-compatible prediction column name
#   • "P (<target>=<class>)" per class       ← when outputClassProbabilities is true
#
# - Inputs: one table with a target column (classColumn) plus feature columns.
# - Feature selection: all numeric/boolean columns except the target. Values are coerced to
#   numeric (invalid → NaN) and filled with 0.0 to satisfy KNN distance computations.
# - Hyperparameters: k (neighbors), weightByDistance → weights ('uniform'|'distance').
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (
    first,
    first_el,
    normalize_in_ports,
    collect_module_imports,
    split_out_imports,
)

# KNIME factory for K Nearest Neighbor (v2)
FACTORY = "org.knime.base.node.mine.knn.KnnNodeFactory2"

# ---------------------------------------------------------------------
# settings.xml → KNNSettings
# ---------------------------------------------------------------------

@dataclass
class KNNSettings:
    target_col: Optional[str] = None
    k: int = 3
    weight_by_distance: bool = False
    output_probs: bool = True


def _bool(s: Optional[str], default: bool) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def _to_int(s: Optional[str], default: int) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return default


def parse_knn_settings(node_dir: Optional[Path]) -> KNNSettings:
    if not node_dir:
        return KNNSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return KNNSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return KNNSettings()

    target = first(model_el, ".//*[local-name()='entry' and @key='classColumn']/@value")
    k_val = _to_int(first(model_el, ".//*[local-name()='entry' and @key='k']/@value"), 3)
    w_by_dist = _bool(first(model_el, ".//*[local-name()='entry' and @key='weightByDistance']/@value"), False)
    out_probs = _bool(first(model_el, ".//*[local-name()='entry' and @key='outputClassProbabilities']/@value"), True)

    return KNNSettings(
        target_col=(target or None),
        k=max(int(k_val), 1),
        weight_by_distance=bool(w_by_dist),
        output_probs=bool(out_probs),
    )

# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    return [
        "import pandas as pd",
        "from sklearn.neighbors import KNeighborsClassifier",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.knn.KnnNodeFactory2"
)

def _emit_knn_code(cfg: KNNSettings) -> List[str]:
    lines: List[str] = []
    lines.append(f"_target = {repr(cfg.target_col) if cfg.target_col else 'None'}")
    lines.append(f"_k = int({cfg.k})")
    w = "distance" if cfg.weight_by_distance else "uniform"
    lines.append(f"_weights = {w!r}")
    lines.append(f"_emit_probs = {('True' if cfg.output_probs else 'False')}")
    lines.append("")
    lines.append("df = context[data_key]")
    lines.append("out_df = df.copy()")
    lines.append("")
    lines.append("# Validate target presence")
    lines.append("if _target is None or _target not in out_df.columns:")
    lines.append("    raise KeyError(f'KNN: target column not found: {_target!r}')")
    lines.append("")
    lines.append("# Feature selection: use all numeric/boolean columns except the target")
    lines.append("cand = [c for c in out_df.columns if c != _target]")
    lines.append("num_like = out_df[cand].select_dtypes(include=['number','bool','boolean','Int64','Float64']).columns.tolist()")
    lines.append("if not num_like:")
    lines.append("    # More permissive fallback: coerce everything to numeric and keep non-all-NaN")
    lines.append("    X_try = out_df[cand].apply(pd.to_numeric, errors='coerce')")
    lines.append("    num_like = [c for c in cand if not X_try[c].isna().all()]")
    lines.append("    if not num_like:")
    lines.append("        raise ValueError('KNN: no usable (numeric) feature columns found')")
    lines.append("X = out_df[num_like].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)")
    lines.append("y = out_df[_target]")
    lines.append("")
    lines.append("# Fit & score")
    lines.append("model = KNeighborsClassifier(n_neighbors=_k, weights=_weights)")
    lines.append("model.fit(X, y)")
    lines.append("pred = model.predict(X)")
    # KNIME-compatible prediction column name:
    lines.append("out_df['Class [kNN]'] = pd.Series(pred, index=out_df.index).astype('object')")
    lines.append("")
    lines.append("# Probabilities (optional)")
    lines.append("if _emit_probs and hasattr(model, 'predict_proba'):")
    lines.append("    proba = model.predict_proba(X)")
    lines.append("    classes = list(getattr(model, 'classes_', []))  # sklearn labels")
    lines.append("    for j, cls in enumerate(classes):")
    lines.append("        cname = f\"P ({_target}={cls})\"")
    lines.append("        out_df[cname] = proba[:, j]")
    lines.append("")
    lines.append("# Publish scored table")
    lines.append("context[out_port_key] = out_df")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],        # Port 1 = data table
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_knn_settings(ndir)

    # Resolve the single input
    pairs = normalize_in_ports(in_ports)
    data_src, data_in = pairs[0] if pairs else ("UNKNOWN", "1")

    # Resolve output port id(s); default to "1"
    ports = out_ports or ["1"]
    out_port = sorted({(p or "1") for p in ports})[0]

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"data_key = '{data_src}:{data_in}'")
    lines.append(f"out_port_key = '{node_id}:{out_port}'")
    lines.extend(_emit_knn_code(cfg))
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

    Port mapping:
      - Input 1 → data table
      - Output 1 → scored table (predictions and optional probabilities)
    """
    explicit_imports = collect_module_imports(generate_imports)

    # Normalize inputs to (src_id, src_port)
    norm_in = [(str(src), str(getattr(e, "source_port", "") or "1")) for src, e in (incoming or [])]
    # Gather output ports (usually just "1")
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, norm_in, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
