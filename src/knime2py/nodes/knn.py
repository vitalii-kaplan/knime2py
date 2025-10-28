#!/usr/bin/env python3

####################################################################################################
#
# K Nearest Neighbor (trainer + scorer)
#
# Behavior:
#   • If TWO inputs are connected:
#       Port 1 → TRAIN table (used to fit)
#       Port 2 → TEST  table (scored output)
#     (Mapping is done via the incoming edge's target_port; order doesn't matter.)
#   • If ONE input is connected:
#       It is used as both train and test (fit + in-sample score).
#
# Output columns:
#   • "Class [kNN]"                          ← KNIME-compatible prediction column name
#   • "P (<target>=<class>)" per class       ← when outputClassProbabilities is true
#
# Feature selection:
#   • From the TRAIN table: all numeric/boolean columns EXCEPT the target and any prediction/prob columns.
#   • Test is reindexed to the same feature order; missing columns are created as 0.0.
#
# Hyperparameters:
#   • k (neighbors), weightByDistance → weights ('uniform'|'distance').
#
####################################################################################################

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


def _to_int(s: Optional[str], default: int) -> int:
    """
    Convert a string to an integer value.

    Args:
        s (Optional[str]): The string to convert.
        default (int): The default value to return if conversion fails.

    Returns:
        int: The converted integer value.
    """
    try:
        return int(str(s).strip())
    except Exception:
        return default


def parse_knn_settings(node_dir: Optional[Path]) -> KNNSettings:
    """
    Parse the KNN settings from the settings.xml file.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        KNNSettings: An instance of KNNSettings populated with values from the XML.
    """
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
    """
    Generate the necessary import statements for the KNN implementation.

    Returns:
        List[str]: A list of import statements.
    """
    return [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.neighbors import KNeighborsClassifier",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.knn.KnnNodeFactory2"
)

def _emit_knn_code(cfg: KNNSettings) -> List[str]:
    """
    Emit the KNN code based on the provided configuration settings.

    Args:
        cfg (KNNSettings): The configuration settings for the KNN.

    Returns:
        List[str]: A list of code lines for the KNN implementation.
    """
    lines: List[str] = []
    lines.append(f"_target = {repr(cfg.target_col) if cfg.target_col else 'None'}")
    lines.append(f"_k = int({cfg.k})")
    w = "distance" if cfg.weight_by_distance else "uniform"
    lines.append(f"_weights = {w!r}")
    lines.append(f"_emit_probs = {('True' if cfg.output_probs else 'False')}")
    lines.append("")
    lines.append("# Resolve train/test tables")
    lines.append("train_df = context[train_key]")
    lines.append("test_df  = context[test_key] if test_key is not None else context[train_key]")
    lines.append("out_df = test_df.copy()")
    lines.append("")
    lines.append("# Validate target presence in TRAIN")
    lines.append("if _target is None or _target not in train_df.columns:")
    lines.append("    raise KeyError(f'KNN: target column not found in train table: {_target!r}')")
    lines.append("")
    lines.append("# ---- Feature selection from TRAIN ----")
    lines.append("cand = [c for c in train_df.columns if c != _target]")
    # Exclude obvious prediction/probability artifacts if present upstream
    lines.append("cand = [c for c in cand if c != 'Class [kNN]' and not str(c).startswith('Prediction (') and not str(c).startswith('P (')]")
    lines.append("num_like = train_df[cand].select_dtypes(include=['number','bool','boolean','Int64','Float64']).columns.tolist()")
    lines.append("if not num_like:")
    lines.append("    X_try = train_df[cand].apply(pd.to_numeric, errors='coerce')")
    lines.append("    num_like = [c for c in cand if not X_try[c].isna().all()]")
    lines.append("    if not num_like:")
    lines.append("        raise ValueError('KNN: no usable (numeric) feature columns found in train table')")
    lines.append("features = list(num_like)")
    lines.append("")
    lines.append("# ---- Build TRAIN matrices ----")
    lines.append("X_tr = train_df[features].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)")
    lines.append("y_tr = train_df[_target].astype('object')")
    lines.append("")
    lines.append("# ---- Align TEST to TRAIN feature space ----")
    lines.append("for c in features:")
    lines.append("    if c not in test_df.columns:")
    lines.append("        # create missing test columns as 0.0 (neutral for distance)")
    lines.append("        out_df[c] = 0.0")
    lines.append("X_te = out_df.reindex(columns=features).apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)")
    lines.append("")
    lines.append("# ---- Fit & score ----")
    lines.append("model = KNeighborsClassifier(n_neighbors=_k, weights=_weights)")
    lines.append("model.fit(X_tr, y_tr)")
    lines.append("pred = model.predict(X_te)")
    lines.append("out_df['Class [kNN]'] = pd.Series(pred, index=out_df.index).astype('object')")
    lines.append("")
    lines.append("# Probabilities (optional)")
    lines.append("if _emit_probs and hasattr(model, 'predict_proba'):")
    lines.append("    try:")
    lines.append("        proba = model.predict_proba(X_te)")
    lines.append("        classes = list(getattr(model, 'classes_', []))  # sklearn class labels")
    lines.append("        for j, cls in enumerate(classes):")
    lines.append("            cname = f\"P ({_target}={cls})\"")
    lines.append("            out_df[cname] = proba[:, j]")
    lines.append("    except Exception as _e:")
    lines.append("        # Soft-fail on probability emission; keep predictions")
    lines.append("        pass")
    lines.append("")
    lines.append("# Publish scored table (TEST rows with predictions)")
    lines.append("context[out_port_key] = out_df")
    return lines


def _order_incoming_by_target_port(in_ports) -> List[Tuple[int, str, str]]:
    """
    Order incoming ports by their target port index.

    Args:
        in_ports: A list of incoming ports.

    Returns:
        List[Tuple[int, str, str]]: A list of tuples containing the target port index, source ID, and source port.
    """
    ordered: List[Tuple[int, str, str]] = []
    for src_id, e in (in_ports or []):
        try:
            t_raw = getattr(e, "target_port", None)
            t_idx = int(t_raw) if t_raw is not None and str(t_raw).strip().isdigit() else 999999
        except Exception:
            t_idx = 999999
        src_port = str(getattr(e, "source_port", "") or "1")
        ordered.append((t_idx, str(src_id), src_port))
    if not ordered:
        # fallback to simple normalization (no target port info)
        pairs = normalize_in_ports(in_ports)
        ordered = [(i + 1, sid, sp) for i, (sid, sp) in enumerate(pairs)]
    ordered.sort(key=lambda t: (t[0], t[1], t[2]))
    return ordered


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],        # may have 1 or 2 table inputs
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python code body for the KNN node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: A list of code lines for the KNN implementation.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_knn_settings(ndir)

    # Determine train/test by KNIME target ports (1=train, 2=test). Order-insensitive.
    ordered = _order_incoming_by_target_port(in_ports)

    if len(ordered) >= 2:
        # Prefer explicit port mapping when possible
        train_tuple = next((t for t in ordered if t[0] == 1), ordered[0])
        test_tuple  = next((t for t in ordered if t[0] == 2), ordered[1])
    elif len(ordered) == 1:
        train_tuple = test_tuple = ordered[0]  # fit & score on the same table
    else:
        # No inputs wired — defensive default keys
        train_tuple = test_tuple = (1, "UNKNOWN", "1")

    _, tr_src, tr_port = train_tuple
    _, te_src, te_port = test_tuple

    # Resolve output port id(s); default to "1"
    ports = out_ports or ["1"]
    out_port = sorted({(p or "1") for p in ports})[0]

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"train_key = '{tr_src}:{tr_port}'")
    # If test is same as train, we still set test_key identically; logic above handles it.
    lines.append(f"test_key = '{te_src}:{te_port}'")
    lines.append(f"out_port_key = '{node_id}:{out_port}'")
    lines.extend(_emit_knn_code(cfg))
    return lines


def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    """
    Generate the code for a Jupyter notebook cell for the KNN node.

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
    Handle the KNN node and generate the necessary imports and body lines.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming edges.
        outgoing: The outgoing edges.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines.
    """
    explicit_imports = collect_module_imports(generate_imports)

    # Preserve incoming edge objects to read target_port reliably
    in_ports = [(str(src), e) for src, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
