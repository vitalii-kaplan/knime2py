#!/usr/bin/env python3

####################################################################################################
#
# X-Partitioner
#
# Module-level contract:
#   - LOOP = "start"  (used by emitter.py to recognize loop start nodes)
# Generated code behavior (no LOOP variable emitted into the script):
#   - Reads input df from context and freezes it in __xpart_src_<node_id>.
#   - Builds cross-validation folds from settings.xml:
#       * validations (k), leaveOneOut, stratifiedSampling, randomSampling,
#         classColumn, useRandomSeed, randomSeed
#       * StratifiedKFold / KFold / LeaveOneOut
#       * shuffle == randomSampling; random_state honored only with shuffle=True
#       * NaN class labels treated as "__NA__" for stratification
#   - Saves loop state in context['__loop__:<node_id>'] and flowvars.
#
#   - Emits a Python `for` loop:
#         for __fold_idx, (__tr, __te) in enumerate(folds):
#     Inside it sets train/test and per-iteration flowvars.
#   - The emitter will indent subsequent nodes until the X-Aggregator (LOOP="finish").
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# Signal to emitter: this node starts a loop
LOOP = "start"

# KNIME factory id (must match settings.xml)
FACTORY = "org.knime.base.node.meta.xvalidation.XValidatePartitionerFactory"

# --------------------------------------------------------------------------------------------------
# settings.xml → Settings
# --------------------------------------------------------------------------------------------------

@dataclass
class XPartSettings:
    k: int = 10
    random_sampling: bool = False
    leave_one_out: bool = False
    stratified: bool = True
    class_col: Optional[str] = None
    use_seed: bool = False
    seed: Optional[int] = None


def _bool(v: Optional[str], default: bool) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def _to_int(v: Optional[str], default: int) -> int:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _to_long(v: Optional[str], default: Optional[int]) -> Optional[int]:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def parse_xpart_settings(node_dir: Optional[Path]) -> XPartSettings:
    if not node_dir:
        return XPartSettings()

    sp = node_dir / "settings.xml"
    if not sp.exists():
        return XPartSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")

    k = _to_int(first(model, ".//*[local-name()='entry' and @key='validations']/@value"), 10) if model is not None else 10
    rnd = _bool(first(model, ".//*[local-name()='entry' and @key='randomSampling']/@value"), False) if model is not None else False
    loo = _bool(first(model, ".//*[local-name()='entry' and @key='leaveOneOut']/@value"), False) if model is not None else False
    strat = _bool(first(model, ".//*[local-name()='entry' and @key='stratifiedSampling']/@value"), True) if model is not None else True
    cls = first(model, ".//*[local-name()='entry' and @key='classColumn']/@value") if model is not None else None
    use_seed = _bool(first(model, ".//*[local-name()='entry' and @key='useRandomSeed']/@value"), False) if model is not None else False
    seed = _to_long(first(model, ".//*[local-name()='entry' and @key='randomSeed']/@value"), 0) if (model is not None and use_seed) else None

    if not loo and k < 2:
        k = 2

    return XPartSettings(
        k=k,
        random_sampling=rnd,
        leave_one_out=loo,
        stratified=strat,
        class_col=(cls or None),
        use_seed=use_seed,
        seed=seed if rnd else None,  # seed used only when shuffle/randomSampling is True
    )

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    return [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.meta.xvalidation.XValidatePartitionerFactory"
)


def _emit_xpart_code(cfg: XPartSettings, node_id: str, src_var: str) -> list[str]:
    """
    Emit loop setup and iteration using `src_var` as the frozen source DataFrame.
    """
    lines: list[str] = []
    lines.append("# Plan cross-validation folds and open the loop")
    lines.append(f"_leave_one_out = {bool(cfg.leave_one_out)}")
    lines.append(f"_k = {('None' if cfg.leave_one_out else str(max(2, int(cfg.k))))}")
    lines.append(f"_shuffle = {bool(cfg.random_sampling)}")
    lines.append(f"_seed = {repr(cfg.seed)}")
    lines.append(f"_stratified = {bool(cfg.stratified)}")
    lines.append(f"_class_col = {repr(cfg.class_col) if cfg.class_col else 'None'}")
    lines.append("")
    lines.append("_y = None")
    lines.append(f"if _stratified and _class_col and _class_col in {src_var}.columns:")
    lines.append(f"    _y = {src_var}[_class_col].astype('object').where(pd.notna({src_var}[_class_col]), '__NA__')")
    lines.append("")
    lines.append("folds = []  # list of (train_idx, test_idx)")
    lines.append("if _leave_one_out:")
    lines.append("    splitter = LeaveOneOut()")
    lines.append(f"    for tr, te in splitter.split({src_var}, _y if _y is not None else None):")
    lines.append("        folds.append((tr, te))")
    lines.append("else:")
    lines.append("    if _stratified and _y is not None:")
    lines.append("        splitter = StratifiedKFold(n_splits=int(_k), shuffle=_shuffle, random_state=_seed if _shuffle else None)")
    lines.append(f"        for tr, te in splitter.split({src_var}, _y):")
    lines.append("            folds.append((tr, te))")
    lines.append("    else:")
    lines.append("        splitter = KFold(n_splits=int(_k), shuffle=_shuffle, random_state=_seed if _shuffle else None)")
    lines.append(f"        for tr, te in splitter.split({src_var}):")
    lines.append("            folds.append((tr, te))")
    lines.append("")
    lines.append("if not folds:")
    lines.append(f"    folds = [(np.arange(len({src_var})), np.array([], dtype=int))]")
    lines.append("")
    lines.append("loop_state = {")
    lines.append(f"    'node_id': {repr(node_id)},")
    lines.append("    'k': len(folds),")
    lines.append("    'current': 0,")
    lines.append("    'classColumn': _class_col,")
    lines.append("    'stratified': bool(_y is not None),")
    lines.append("    'shuffle': _shuffle,")
    lines.append("    'seed': _seed,")
    lines.append("    'folds': folds,")
    lines.append("}")
    lines.append(f"context['__loop__:{node_id}'] = loop_state")
    lines.append(f"context['__flowvar__:{node_id}:maxIterations'] = len(folds)")
    lines.append("")
    lines.append("for __fold_idx, (__tr, __te) in enumerate(folds):")
    lines.append(f"    context['__flowvar__:{node_id}:currentIteration'] = int(__fold_idx)")
    lines.append(f"    context['__loop__:{node_id}']['current'] = int(__fold_idx)")
    # IMPORTANT: index from the frozen source df, not 'df'
    lines.append(f"    train_df = {src_var}.iloc[__tr].reset_index(drop=True)")
    lines.append(f"    test_df  = {src_var}.iloc[__te].reset_index(drop=True)")
    lines.append(f"    context['{node_id}:1'] = train_df")
    lines.append(f"    context['{node_id}:2'] = test_df")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_xpart_settings(ndir)

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    # Freeze the loop source DF in a unique variable that won't be clobbered by inner nodes
    src_var = f"__xpart_src_{node_id}"
    lines.append(f"{src_var} = context['{src_id}:{in_port}']  # input table (frozen for the whole CV loop)")

    # Emit loop body using the frozen variable
    lines.extend(_emit_xpart_code(cfg, node_id, src_var))
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
    Returns (imports, body_lines) if this module can handle the node; else None.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports  = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
