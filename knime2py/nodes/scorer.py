#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory for Scorer
SCORER_FACTORY = "org.knime.base.node.mine.scorer.accuracy.AccuracyScorer2NodeFactory"


def can_handle(node_type: Optional[str]) -> bool:
    return bool(node_type and node_type.endswith(SCORER_FACTORY))


# ---------------------------------------------------------------------
# settings.xml → ScorerSettings
# ---------------------------------------------------------------------

@dataclass
class ScorerSettings:
    truth_col: str = "class"                   # KNIME: entry key="first"
    pred_col: str = "Prediction (class)"       # KNIME: entry key="second"
    ignore_missing: bool = True                # KNIME: entry key="ignore.missing.values"


def _bool(v: Optional[str], default: bool) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def parse_scorer_settings(node_dir: Optional[Path]) -> ScorerSettings:
    if not node_dir:
        return ScorerSettings()
    sp = node_dir / "settings.xml"
    if not sp.exists():
        return ScorerSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return ScorerSettings()

    truth = first(model, ".//*[local-name()='entry' and @key='first']/@value") or "class"
    pred = first(model, ".//*[local-name()='entry' and @key='second']/@value") or f"Prediction ({truth})"
    ign = _bool(first(model, ".//*[local-name()='entry' and @key='ignore.missing.values']/@value"), True)

    return ScorerSettings(truth_col=truth, pred_col=pred, ignore_missing=ign)


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    return [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.scorer.accuracy.AccuracyScorer2NodeFactory"
)


def _emit_scorer_code(cfg: ScorerSettings) -> List[str]:
    lines: List[str] = []
    lines.append("out_df = df.copy()  # not strictly required, but keeps pattern consistent")
    lines.append(f"_truth_col = {repr(cfg.truth_col)}")
    lines.append(f"_pred_col  = {repr(cfg.pred_col)}")
    lines.append("")
    lines.append("# Validate required columns exist")
    lines.append("missing = []")
    lines.append("if _truth_col not in df.columns: missing.append(_truth_col)")
    lines.append("if _pred_col not in df.columns:  missing.append(_pred_col)")
    lines.append("if missing:")
    lines.append("    raise KeyError(f'Scorer: missing required column(s): {missing}')")
    lines.append("")
    # Select, optionally drop NA rows
    lines.append("pair = df[[_truth_col, _pred_col]].copy()")
    if cfg.ignore_missing:
        lines.append("pair = pair.dropna(subset=[_truth_col, _pred_col])  # KNIME: ignore.missing.values = true")
    else:
        lines.append("# KNIME: ignore.missing.values = false (keep rows with NA) — sklearn metrics will error on NA")
        lines.append("# You may want to coerce NA to a sentinel if needed:")
        lines.append("# pair = pair.fillna({'_truth_col': '__NA__', '_pred_col': '__NA__'})")
    lines.append("")
    lines.append("y_true = pair[_truth_col].astype('object')")
    lines.append("y_pred = pair[_pred_col].astype('object')")
    lines.append("")
    lines.append("# Metrics")
    lines.append("acc = float(accuracy_score(y_true, y_pred)) if len(pair) else float('nan')")
    lines.append("err = (1.0 - acc) if pd.notna(acc) else float('nan')")
    lines.append("correct = int((y_true == y_pred).sum())")
    lines.append("false   = int((y_true != y_pred).sum())")
    lines.append("try:")
    lines.append("    kappa = float(cohen_kappa_score(y_true, y_pred)) if len(pair) else float('nan')")
    lines.append("except Exception:")
    lines.append("    kappa = float('nan')")
    lines.append("")
    # Confusion matrix
    lines.append("# Confusion matrix with a stable label list (union from both columns, order of appearance)")
    lines.append("labels = pd.unique(pd.concat([y_true, y_pred], ignore_index=True))")
    lines.append("if len(labels) == 0:")
    lines.append("    cm_df = pd.DataFrame(columns=pd.Index([], name='Predicted'), index=pd.Index([], name='Actual'))")
    lines.append("else:")
    lines.append("    cm = confusion_matrix(y_true, y_pred, labels=labels)")
    lines.append("    cm_df = pd.DataFrame(cm, index=pd.Index(labels, name='Actual'), columns=pd.Index(labels, name='Predicted'))")
    lines.append("")
    # Summary table (match KNIME-like fields)
    lines.append("summary_rows = [")
    lines.append("    {'Metric': 'Accuracy',       'Value': acc},")
    lines.append("    {'Metric': 'Error',          'Value': err},")
    lines.append("    {'Metric': '#Correct',       'Value': correct},")
    lines.append("    {'Metric': '#False',         'Value': false},")
    lines.append("    {'Metric': \"Cohen's kappa\", 'Value': kappa},")
    lines.append("]")
    lines.append("summary_df = pd.DataFrame(summary_rows)")
    lines.append("")
    lines.append("# Outputs:")
    lines.append("# Port 1 → confusion matrix, Port 2 → summary metrics")
    lines.append("cm_out = cm_df")
    lines.append("stats_out = summary_df")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_scorer_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table with truth & prediction")

    lines.extend(_emit_scorer_code(cfg))

    # Publish to context — ensure two distinct ports; confusion on the smaller port id
    ports = [str(p or "1") for p in (out_ports or ["1", "2"])]
    if len(ports) == 1:
        ports.append("2")
    ports = list(dict.fromkeys(ports))
    if len(ports) == 1:
        ports.append("2" if ports[0] != "2" else "1")

    def _port_key(p: str):
        return (0, int(p)) if p.isdigit() else (1, p)

    p1, p2 = sorted(ports, key=_port_key)[:2]
    lines.append(f"context['{node_id}:{p1}'] = cm_out")
    lines.append(f"context['{node_id}:{p2}'] = stats_out")

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
    if not (ntype and can_handle(ntype)):
        return None

    explicit_imports = collect_module_imports(generate_imports)

    # One input (scored table)
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])] or [("UNKNOWN", "1")]

    # Two outputs (confusion matrix, stats); ports may be connected in any order
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
