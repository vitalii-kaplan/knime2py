#!/usr/bin/env python3

####################################################################################################
#
# SVM Learner
#
# Trains a scikit-learn SVC from parameters parsed in settings.xml and emits:
# (1) a model bundle (estimator + metadata), (2) a coefficient table (when available), and (3) a
# summary. Mapping (KNIME → sklearn):
#   - Kernel:
#       • "RBF"               → kernel='rbf',   gamma from Sigma as gamma = 1/(2*sigma^2)
#       • "Polynomial"        → kernel='poly',  degree from Power, coef0 from Bias, gamma from Gamma
#       • "HyperTangent"      → kernel='sigmoid', gamma from Kappa, coef0 from Delta
#       • "Linear" (if present) → kernel='linear'
#   - Cost parameter C (variously named "C" / "cost" / "regularization") → C
#   - Probability: enabled (probability=True) to support downstream ROC/probabilities.
#
#   - Feature coefficients only exist for linear/separable cases; for non-linear kernels we emit an
#     empty coefficient table.
#   - Scaling is not applied here; if KNIME’s node performs internal scaling, replicate upstream.
#   - Random seed: SVC uses it for probability calibration; default to 1 for reproducibility.
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, normalize_in_ports, collect_module_imports, split_out_imports, iter_entries

# KNIME factory for SVM Learner (Base)
FACTORY = "org.knime.base.node.mine.svm.learner.SVMLearnerNodeFactory2"

# Hub doc (kept as a reference)
HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.svm.learner.SVMLearnerNodeFactory2"
)

# ---------------------------------------------------------------------
# settings.xml → SVMLearnerSettings
# ---------------------------------------------------------------------

@dataclass
class SVMLearnerSettings:
    target: Optional[str] = None
    kernel: str = "rbf"             # 'linear' | 'rbf' | 'poly' | 'sigmoid'
    C: float = 1.0
    degree: int = 3                 # poly only (KNIME: Power)
    gamma: str | float = "scale"    # 'scale' | 'auto' | float
    coef0: float = 0.0              # poly/sigmoid bias/Delta
    probability: bool = True
    class_weight: Optional[str] = None  # e.g., 'balanced'
    random_state: int = 1

# ----------------------------
# Local parsing helpers (robust over varied key names)
# ----------------------------

def _tokfind(entries: Dict[str, str], *tokens: str) -> Optional[str]:
    """Return first value whose key contains ALL tokens (case-insensitive)."""
    toks = [t.lower() for t in tokens if t]
    for k, v in entries.items():
        lk = k.lower()
        if all(t in lk for t in toks):
            return v
    return None

def _to_float(v: Optional[str], default: float | None = None) -> Optional[float]:
    try:
        return float(v) if v is not None and v != "" else default
    except Exception:
        return default

def _to_int(v: Optional[str], default: int | None = None) -> Optional[int]:
    try:
        return int(v) if v is not None and v != "" else default
    except Exception:
        return default

def _to_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}

def _collect_entries(el: ET._Element) -> Dict[str, str]:
    """Flatten <entry key=... value=.../> into a dict (first win)."""
    out: Dict[str, str] = {}
    for k, v in iter_entries(el):
        if k not in out:
            out[k] = v or ""
    return out

# KNIME → sklearn kernel mapping (accepts several synonyms)
_KERNEL_MAP = {
    "RBF": "rbf",
    "GAUSSIAN": "rbf",
    "LINEAR": "linear",
    "POLYNOMIAL": "poly",
    "POLY": "poly",
    "HYPERTANGENT": "sigmoid",   # KNIME naming
    "SIGMOID": "sigmoid",
}

def _sigma_to_gamma(sigma: Optional[float]) -> Optional[float]:
    if sigma is None or sigma <= 0:
        return None
    return 1.0 / (2.0 * sigma * sigma)

# ----------------------------
# Parse settings.xml
# ----------------------------

def parse_svm_settings(node_dir: Optional[Path]) -> SVMLearnerSettings:
    if not node_dir:
        return SVMLearnerSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return SVMLearnerSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return SVMLearnerSettings()

    ent = _collect_entries(model_el)

    # Target column
    target = (
        first(model_el, ".//*[local-name()='entry' and @key='classifyColumn']/@value")
        or _tokfind(ent, "class", "column")  # fallback
    )

    # Kernel
    raw_kernel = (
        _tokfind(ent, "kernel")
        or first(model_el, ".//*[local-name()='entry' and @key='kernel']/@value")
        or "RBF"
    )
    sk_kernel = _KERNEL_MAP.get((raw_kernel or "").strip().upper(), "rbf")

    # Cost parameter (C)
    raw_C = (
        ent.get("C")
        or _tokfind(ent, "cost")
        or _tokfind(ent, "regularization")
        or _tokfind(ent, "penalty")
    )
    C = _to_float(raw_C, 1.0) or 1.0

    # Probability
    prob = _to_bool(_tokfind(ent, "prob"), True)

    # Class weight (optional)
    cw = _tokfind(ent, "class", "weight")
    class_weight = "balanced" if isinstance(cw, str) and cw.strip().lower() == "balanced" else None

    # Kernel-specific params
    degree = 3
    gamma: str | float = "scale"
    coef0 = 0.0

    if sk_kernel == "rbf":
        # KNIME uses Sigma; sklearn expects gamma
        sigma = _to_float(_tokfind(ent, "sigma"))
        g = _sigma_to_gamma(sigma)
        if g is not None:
            gamma = g

    elif sk_kernel == "poly":
        # Polynomial(Power, Bias, Gamma)
        degree = _to_int(_tokfind(ent, "power"), 3) or 3
        coef0 = _to_float(_tokfind(ent, "bias"), 0.0) or 0.0
        g = _to_float(_tokfind(ent, "gamma"))
        if g is not None:
            gamma = g  # direct

    elif sk_kernel == "sigmoid":
        # HyperTangent(Kappa, Delta) ~ sigmoid(gamma=Kappa, coef0=Delta)
        g = _to_float(_tokfind(ent, "kappa"))
        if g is not None:
            gamma = g
        d = _to_float(_tokfind(ent, "delta"))
        if d is not None:
            coef0 = d

    # linear: no degree/gamma/coef0 needed

    return SVMLearnerSettings(
        target=target or None,
        kernel=sk_kernel,
        C=float(C),
        degree=int(degree),
        gamma=gamma,
        coef0=float(coef0),
        probability=bool(prob),
        class_weight=class_weight,
        random_state=1,
    )

# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    return [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.svm import SVC",
    ]

def _emit_train_code(cfg: SVMLearnerSettings) -> List[str]:
    """
    Emit lines that:
      - select X, y (numeric/bool features by default, excluding target)
      - fit sklearn SVC
      - produce coefficient table (if available) & a summary
      - build a *bundle* dict with estimator + metadata for downstream Predictor
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    if not cfg.target:
        lines.append("# No target column configured; passthrough / no model.")
        lines.append("model = None")
        lines.append("coef_df = pd.DataFrame(columns=['feature','coef'])")
        lines.append("summary_df = pd.DataFrame([{'error': 'no target configured'}])")
        lines.append("bundle = {'estimator': None, 'features': [], 'feature_cols': [], 'target': None, 'classes': []}")
        return lines

    # Target
    lines.append(f"_target = {repr(cfg.target)}")

    # Feature selection: numeric/boolean, excluding target
    lines.append("num_like = df.select_dtypes(include=['number','bool','boolean','Int64','Float64']).columns.tolist()")
    lines.append("feat_cols = [c for c in num_like if c != _target]")

    # Extract X, y
    lines.append("X = df[feat_cols].copy()")
    lines.append("y = df[_target].copy()")

    # Hyperparameters
    lines.append(f"_kernel = {repr(cfg.kernel)}")
    lines.append(f"_C = float({float(cfg.C)})")
    if isinstance(cfg.gamma, str):
        lines.append(f"_gamma = {repr(cfg.gamma)}")
    else:
        lines.append(f"_gamma = float({float(cfg.gamma)})")
    lines.append(f"_degree = int({int(cfg.degree)})")
    lines.append(f"_coef0 = float({float(cfg.coef0)})")
    lines.append(f"_probability = bool({bool(cfg.probability)})")
    lines.append(f"_class_weight = {repr(cfg.class_weight)}")
    lines.append(f"_seed = int({int(cfg.random_state)})  # used for probability calibration")

    # Fit model
    lines.append("model = SVC(")
    lines.append("    kernel=_kernel,")
    lines.append("    C=_C,")
    lines.append("    gamma=_gamma,")
    lines.append("    degree=_degree,")
    lines.append("    coef0=_coef0,")
    lines.append("    probability=_probability,")
    lines.append("    class_weight=_class_weight,")
    lines.append("    random_state=_seed,")
    lines.append(")")
    lines.append("model.fit(X, y)")

    # Coefficients (if available)
    lines.append("coef_df = pd.DataFrame(columns=['feature','coef'])")
    lines.append("if hasattr(model, 'coef_') and model.coef_ is not None:")
    lines.append("    try:")
    lines.append("        # For linear kernel and one-vs-one, flatten per-pair coefs")
    lines.append("        coeffs = model.coef_")
    lines.append("        if coeffs.ndim == 1:")
    lines.append("            coef_df = pd.DataFrame({'feature': feat_cols, 'coef': coeffs})")
    lines.append("        else:")
    lines.append("            # stack rows with pair index")
    lines.append("            tmp = []")
    lines.append("            for i, row in enumerate(coeffs):")
    lines.append("                tmp.append(pd.DataFrame({'feature': feat_cols, 'coef': row, 'pair': i}))")
    lines.append("            coef_df = pd.concat(tmp, ignore_index=True)")
    lines.append("        coef_df = coef_df.sort_values(by='coef', key=lambda s: s.abs(), ascending=False, kind='mergesort').reset_index(drop=True)")
    lines.append("    except Exception:")
    lines.append("        coef_df = pd.DataFrame(columns=['feature','coef'])")

    # Summary
    lines.append(
        "summary_df = pd.DataFrame([{'kernel': _kernel, 'C': _C, 'gamma': _gamma, "
        "'degree': _degree, 'coef0': _coef0, 'n_features': len(feat_cols), "
        "'classes': ','.join(map(str, getattr(model, 'classes_', [])))}])"
    )

    # Bundle for downstream Predictor
    lines.append("bundle = {")
    lines.append("    'estimator': model,")
    lines.append("    'model': model,  # alias")
    lines.append("    'features': list(feat_cols),")
    lines.append("    'feature_cols': list(feat_cols),")
    lines.append("    'target': _target,")
    lines.append("    'classes': list(getattr(model, 'classes_', [])),")
    lines.append("    'kernel': _kernel,")
    lines.append("    'C': _C,")
    lines.append("    'gamma': _gamma,")
    lines.append("    'degree': _degree,")
    lines.append("    'coef0': _coef0,")
    lines.append("}")
    return lines

def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_svm_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Input (single table)
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    # Training + outputs
    lines.extend(_emit_train_code(cfg))

    # Publish:
    # Convention: 1=model bundle, 2=coefficients (if any), 3=summary
    ports = [str(p or '1') for p in (out_ports or ['1', '2', '3'])]
    if not ports:
        ports = ['1', '2', '3']

    want = {'1': 'bundle', '2': 'coef_df', '3': 'summary_df'}
    remaining_vals = ['bundle', 'coef_df', 'summary_df']
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
    Returns (imports, body_lines).
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2", "3"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
