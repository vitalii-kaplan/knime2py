#!/usr/bin/env python3

####################################################################################################
#
# SVM Learner  — fail-fast if misconfigured (reads KNIME keys: classcol, kernel_type, c_parameter,
#                                              kernel_param_* such as Power/Bias/Gamma/Kappa/Delta/Sigma)
#
# Trains a scikit-learn SVC and emits:
#   Port 1: bundle (dict with estimator + metadata)
#   Port 2: coefficients table (empty for non-linear kernels)
#   Port 3: summary table
#
# KNIME → sklearn mapping:
#   kernel_type:
#     - "RBF"          → kernel='rbf',     gamma = 1/(2*sigma^2) from kernel_param_sigma (if present)
#     - "Polynomial"   → kernel='poly',    degree from kernel_param_Power, coef0 from kernel_param_Bias,
#                                           gamma from kernel_param_Gamma (else 'scale')
#     - "HyperTangent" → kernel='sigmoid', gamma from kernel_param_kappa, coef0 from kernel_param_delta
#     - "Linear"       → kernel='linear'
#
#   C parameter: c_parameter → C
#
# Notes:
#   - If no target is configured → raise (fail-fast).
#   - Rows with missing target are dropped prior to fit.
#   - Probability=True to enable downstream probabilities (KNIME predictor can use them).
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

# Hub doc (reference)
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
    degree: int = 3                 # poly only
    gamma: str | float = "scale"    # 'scale' | 'auto' | float
    coef0: float = 0.0              # poly/sigmoid bias/delta
    probability: bool = True
    class_weight: Optional[str] = None
    random_state: int = 1

# Helpers ------------------------------------------------------------------------------------------

def _to_float(v: Optional[str], default: float | None = None) -> Optional[float]:
    """Convert a string to a float, returning a default value if conversion fails."""
    try:
        return float(v) if v is not None and v != "" else default
    except Exception:
        return default

def _to_int(v: Optional[str], default: int | None = None) -> Optional[int]:
    """Convert a string to an integer, returning a default value if conversion fails."""
    try:
        return int(float(v)) if v is not None and v != "" else default
    except Exception:
        return default

def _sigma_to_gamma(sigma: Optional[float]) -> Optional[float]:
    """Convert sigma to gamma for RBF kernel."""
    if sigma is None or sigma <= 0:
        return None
    return 1.0 / (2.0 * sigma * sigma)

_KERNEL_MAP = {
    "RBF": "rbf",
    "GAUSSIAN": "rbf",
    "LINEAR": "linear",
    "POLYNOMIAL": "poly",
    "POLY": "poly",
    "HYPERTANGENT": "sigmoid",
    "SIGMOID": "sigmoid",
}

def _entries_dict(el: ET._Element) -> Dict[str, str]:
    """Extract entries from an XML element into a dictionary."""
    d: Dict[str, str] = {}
    for k, v in iter_entries(el):
        if k not in d:
            d[k] = v or ""
    return d

# Parse settings.xml -------------------------------------------------------------------------------

def parse_svm_settings(node_dir: Optional[Path]) -> SVMLearnerSettings:
    """Parse the settings.xml file and return an SVMLearnerSettings object."""
    if not node_dir:
        return SVMLearnerSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return SVMLearnerSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return SVMLearnerSettings()

    ent = _entries_dict(model_el)

    # Target column (KNIME: classcol)
    target = first(model_el, ".//*[local-name()='entry' and @key='classcol']/@value")

    # Kernel (KNIME: kernel_type)
    raw_kernel = first(model_el, ".//*[local-name()='entry' and @key='kernel_type']/@value") or "RBF"
    sk_kernel = _KERNEL_MAP.get(raw_kernel.strip().upper(), "rbf")

    # Cost parameter (KNIME: c_parameter)
    C = _to_float(first(model_el, ".//*[local-name()='entry' and @key='c_parameter']/@value"), 1.0) or 1.0

    # Kernel-specific parameters
    degree = 3
    gamma: str | float = "scale"
    coef0 = 0.0

    if sk_kernel == "rbf":
        sigma = _to_float(first(model_el, ".//*[local-name()='entry' and @key='kernel_param_sigma']/@value"), None)
        g = _sigma_to_gamma(sigma)
        if g is not None:
            gamma = g
        # if sigma absent/invalid → keep default 'scale'

    elif sk_kernel == "poly":
        # degree from Power (numeric in XML, cast to int); coef0 from Bias; gamma from Gamma
        degree = _to_int(first(model_el, ".//*[local-name()='entry' and @key='kernel_param_Power']/@value"), 3) or 3
        coef0 = _to_float(first(model_el, ".//*[local-name()='entry' and @key='kernel_param_Bias']/@value"), 0.0) or 0.0
        g = _to_float(first(model_el, ".//*[local-name()='entry' and @key='kernel_param_Gamma']/@value"), None)
        if g is not None:
            gamma = g  # otherwise remain 'scale'

    elif sk_kernel == "sigmoid":
        # HyperTangent: Kappa → gamma, Delta → coef0
        g = _to_float(first(model_el, ".//*[local-name()='entry' and @key='kernel_param_kappa']/@value"), None)
        d = _to_float(first(model_el, ".//*[local-name()='entry' and @key='kernel_param_delta']/@value"), None)
        if g is not None:
            gamma = g
        if d is not None:
            coef0 = d

    # Build settings object
    return SVMLearnerSettings(
        target=target or None,
        kernel=sk_kernel,
        C=float(C),
        degree=int(degree),
        gamma=gamma,
        coef0=float(coef0),
        probability=True,          # enable probabilities for downstream predictor
        class_weight=None,         # not present in this XML; could be extended later
        random_state=1,
    )

# Code generators ----------------------------------------------------------------------------------

def generate_imports():
    """Generate import statements for the SVM learner."""
    return [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.svm import SVC",
    ]

def _emit_train_code(cfg: SVMLearnerSettings) -> List[str]:
    """Generate the training code for the SVM model based on the provided configuration."""
    lines: List[str] = []
    lines.append("out_df = df.copy()")

    # --- fail-fast when misconfigured ---
    if not cfg.target:
        lines.append("raise ValueError('SVM Learner: no target configured (settings.xml classcol is missing)')")
        return lines

    # Target present?
    lines.append(f"_target = {repr(cfg.target)}")
    lines.append("if _target not in df.columns:")
    lines.append("    raise KeyError(f\"SVM Learner: target column not found: {_target!r}\")")

    # Feature selection: numeric/bool, excluding target
    lines.append("num_like = df.select_dtypes(include=['number','bool','boolean','Int64','Float64']).columns.tolist()")
    lines.append("feat_cols = [c for c in num_like if c != _target]")
    lines.append("if not feat_cols:")
    lines.append("    raise ValueError('SVM Learner: no numeric/bool feature columns found (after excluding target)')")

    # Extract X, y and drop rows with missing target (sklearn cannot fit with NaN labels)
    lines.append("X = df[feat_cols].copy()")
    lines.append("y = df[_target].copy()")
    lines.append("_mask = y.notna()")
    lines.append("if not bool(_mask.all()):")
    lines.append("    X = X.loc[_mask]")
    lines.append("    y = y.loc[_mask]")
    lines.append("if X.shape[0] == 0:")
    lines.append("    raise ValueError('SVM Learner: no training rows after dropping missing target values')")

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

    # Coefficients (only for linear kernels; scikit exposes coef_ in those cases)
    lines.append("coef_df = pd.DataFrame(columns=['feature','coef'])")
    lines.append("if hasattr(model, 'coef_') and getattr(model, 'coef_', None) is not None:")
    lines.append("    try:")
    lines.append("        coeffs = model.coef_")
    lines.append("        if getattr(coeffs, 'ndim', 1) == 1:")
    lines.append("            coef_df = pd.DataFrame({'feature': feat_cols, 'coef': coeffs})")
    lines.append("        else:")
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
    """Generate the Python body for the SVM learner, including imports and training code."""
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

    # Publish: 1=bundle, 2=coeffs, 3=summary
    ports = [str(p or '1') for p in (out_ports or ['1', '2', '3'])]
    if not ports:
        ports = ['1', '2', '3']
    want = {'1': 'bundle', '2': 'coef_df', '3': 'summary_df'}
    remaining = ['bundle', 'coef_df', 'summary_df']
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
    """Generate the code for a Jupyter notebook cell for the SVM learner."""
    return "\n".join(generate_py_body(node_id, node_dir, in_ports, out_ports)) + "\n"

def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handle the SVM learner node, returning the necessary imports and body lines.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path to the node.
        incoming: Incoming ports.
        outgoing: Outgoing ports.

    Returns:
        A tuple containing the imports and body lines.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2", "3"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
