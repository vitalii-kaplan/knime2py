#!/usr/bin/env python3

####################################################################################################
#
# ROC Curve
#
# Renders ROC curves from a scored table based on settings.xml. Reads ground-truth and positive
# class, resolves one or more probability columns, computes FPR/TPR and AUC, and saves an image
# (PNG/SVG) plus a CSV of ROC points. This view node does not write to context ports.
#
# - Inputs: one table with a truth column and per-class probability columns.
# - Column binding: uses configured target/positive class and selected probability columns; if none
#   are set, attempts to auto-detect KNIME-style columns like "P (<target>=<class>)_LR". Configure
#   explicitly if your suffix differs (e.g., _RF, _GB).
# - Output artifacts: saves "roc_<node_id>.(png|svg)" and "roc_table_<node_id>.csv" in CWD; figure
#   size from width/height (pixels at 100 DPI). Title/axis labels are honored from settings.
# - Implementation: sklearn.metrics.roc_curve/auc for each probability series; matplotlib for
#   plotting; pandas/numpy for data handling.
# 
####################################################################################################


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, iter_entries, normalize_in_ports, collect_module_imports, split_out_imports

# KNIME factory for ROC Curve
FACTORY = "org.knime.base.views.node.roccurve.ROCCurveNodeFactory"

# ---------------------------------------------------------------------
# settings.xml → ROCCurveSettings
# ---------------------------------------------------------------------

@dataclass
class ROCCurveSettings:
    truth_col: Optional[str] = None             # target / ground-truth column
    pos_label: Optional[str] = None             # positive class value (e.g., "yes")
    proba_cols: List[str] = field(default_factory=list)  # selected probability columns
    title: str = "ROC Curve"
    x_label: str = "False positive rate (1 - specificity)"
    y_label: str = "True positive rate (sensitivity)"
    width_px: int = 800
    height_px: int = 600
    image_format: str = "PNG"                   # PNG | SVG (from KNIME; default to PNG)


def _collect_numeric_name_entries(cfg: ET._Element) -> List[str]:
    """Collect <entry key='0' value='...'>, <entry key='1' value='...'> … under cfg."""
    out: List[str] = []
    for k, v in iter_entries(cfg):
        if k.isdigit() and v:
            out.append(v.strip())
    return out


def parse_roc_settings(node_dir: Optional[Path]) -> ROCCurveSettings:
    if not node_dir:
        return ROCCurveSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return ROCCurveSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    # model: render preferences
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    width_px = int(first(model_el, ".//*[local-name()='entry' and @key='width']/@value") or 800) if model_el is not None else 800
    height_px = int(first(model_el, ".//*[local-name()='entry' and @key='height']/@value") or 600) if model_el is not None else 600
    img_fmt = (first(model_el, ".//*[local-name()='entry' and @key='imageFormat']/@value") or "PNG").strip().upper() if model_el is not None else "PNG"

    # view: data bindings and labels
    view_el = first_el(root, ".//*[local-name()='config' and @key='view']")
    truth_col = first(view_el, ".//*[local-name()='config' and @key='targetColumn']/*[local-name()='entry' and @key='selected']/@value") if view_el is not None else None
    pos_label = first(view_el, ".//*[local-name()='entry' and @key='positiveClassValue']/@value") if view_el is not None else None

    title = first(view_el, ".//*[local-name()='entry' and @key='title']/@value") or "ROC Curve"
    x_lab = first(view_el, ".//*[local-name()='entry' and @key='xAxisLabel']/@value") or "False positive rate (1 - specificity)"
    y_lab = first(view_el, ".//*[local-name()='entry' and @key='yAxisLabel']/@value") or "True positive rate (sensitivity)"

    # selected prediction columns
    proba_cols: List[str] = []
    if view_el is not None:
        pred_el = first_el(view_el, ".//*[local-name()='config' and @key='predictionColumns']")
        if pred_el is not None:
            # 1) selected_Internals
            sel_el = first_el(pred_el, ".//*[local-name()='config' and @key='selected_Internals']")
            if sel_el is not None:
                proba_cols.extend(_collect_numeric_name_entries(sel_el))
            # 2) manualFilter/manuallySelected
            man_sel = first_el(pred_el, ".//*[local-name()='config' and @key='manualFilter']/*[local-name()='config' and @key='manuallySelected']")
            if man_sel is not None:
                proba_cols.extend(_collect_numeric_name_entries(man_sel))

    # Deduplicate while preserving order
    proba_cols = list(dict.fromkeys([c for c in proba_cols if c]))

    return ROCCurveSettings(
        truth_col=truth_col or None,
        pos_label=pos_label or None,
        proba_cols=proba_cols,
        title=title,
        x_label=y_lab and x_lab or "False positive rate (1 - specificity)",
        y_label=y_lab or "True positive rate (sensitivity)",
        width_px=width_px,
        height_px=height_px,
        image_format=img_fmt or "PNG",
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    return [
        "from pathlib import Path",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from sklearn.metrics import roc_curve, auc",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.views.node.roccurve.ROCCurveNodeFactory"
)


def _emit_roc_code(cfg: ROCCurveSettings, node_id: str) -> List[str]:
    """
    Emit code that:
      - reads df from context
      - resolves ground-truth, positive label, and probability columns (with fallbacks)
      - computes ROC curves (possibly multiple series)
      - writes ROC figure and CSV with points to the current working directory
    """
    lines: List[str] = []
    lines.append(f"_truth_col = {repr(cfg.truth_col) if cfg.truth_col else 'None'}")
    lines.append(f"_pos_label = {repr(cfg.pos_label) if cfg.pos_label else 'None'}")
    if cfg.proba_cols:
        cols = ", ".join(repr(c) for c in cfg.proba_cols)
        lines.append(f"_proba_cols = [{cols}]")
    else:
        lines.append("_proba_cols = []  # will try to auto-detect probability columns")

    lines.append(f"_title = {repr(cfg.title)}")
    lines.append(f"_x_label = {repr(cfg.x_label)}")
    lines.append(f"_y_label = {repr(cfg.y_label)}")
    lines.append(f"_width_in = {cfg.width_px} / 100.0")
    lines.append(f"_height_in = {cfg.height_px} / 100.0")
    lines.append(f"_img_fmt = {repr((cfg.image_format or 'PNG').upper())}")
    lines.append("")
    lines.append("# Validate inputs")
    lines.append("if _truth_col is None or _truth_col not in df.columns:")
    lines.append("    raise KeyError(f\"ROC: ground-truth column not found: {_truth_col!r}\")")
    lines.append("if _pos_label is None:")
    lines.append("    raise KeyError(\"ROC: positive class value is not configured in settings.xml\")")
    lines.append("")
    lines.append("# Resolve probability columns")
    lines.append("proba_cols = [c for c in _proba_cols if c in df.columns]")
    lines.append("if not proba_cols:")
    lines.append("    # Fallback: guess KNIME-like probability columns produced by our predictor")
    lines.append("    proba_cols = [c for c in df.columns if (c.startswith('P(') or c.startswith('P (')) and c.endswith('_LR')]")
    lines.append("if not proba_cols:")
    lines.append("    raise KeyError('ROC: no probability columns found. Configure them in the node or ensure predictor added P(<class>)_LR columns.')")
    lines.append("")
    lines.append("# Prepare y and iterate curves")
    lines.append("y_true = df[_truth_col].astype('object')")
    lines.append("y_bin = (y_true == _pos_label).astype(int)")
    lines.append("")
    lines.append("rows = []")
    lines.append("for col in proba_cols:")
    lines.append("    # Coerce to [0,1] numeric probabilities")
    lines.append("    p = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float).clip(0.0, 1.0)")
    lines.append("    fpr, tpr, thr = roc_curve(y_bin, p)")
    lines.append("    auc_val = float(auc(fpr, tpr))")
    lines.append("    tmp = pd.DataFrame({'model': col, 'fpr': fpr, 'tpr': tpr, 'threshold': thr})")
    lines.append("    tmp['auc'] = auc_val")
    lines.append("    rows.append(tmp)")
    lines.append("")
    lines.append("points_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['model','fpr','tpr','threshold','auc'])")
    lines.append("")
    lines.append("# Plot")
    lines.append("plt.figure(figsize=(_width_in, _height_in), dpi=100)")
    lines.append("for m, grp in points_df.groupby('model', sort=False):")
    lines.append("    auc_val = float(grp['auc'].iloc[0]) if not grp.empty else float('nan')")
    lines.append("    plt.plot(grp['fpr'].values, grp['tpr'].values, label=f\"{m} (AUC={auc_val:.3f})\")")
    lines.append("plt.plot([0,1], [0,1], linestyle='--', linewidth=1)")
    lines.append("plt.title(_title)")
    lines.append("plt.xlabel(_x_label)")
    lines.append("plt.ylabel(_y_label)")
    lines.append("plt.xlim(0, 1); plt.ylim(0, 1)")
    lines.append("plt.legend(loc='lower right')")
    lines.append("")
    lines.append("out_dir = Path.cwd()")
    lines.append("ext = 'svg' if str(_img_fmt).upper() == 'SVG' else 'png'")
    lines.append(f"img_path = out_dir / ('roc_{node_id}.' + ext)")
    lines.append(f"csv_path = out_dir / ('roc_table_{node_id}.csv')")
    lines.append("plt.savefig(img_path, bbox_inches='tight')")
    lines.append("plt.close()")
    lines.append("points_df.to_csv(csv_path, index=False)")
    lines.append("# Optional: print paths")
    lines.append("print(f'[ROC] Wrote image to: {img_path}')")
    lines.append("print(f'[ROC] Wrote points to: {csv_path}')")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,  # ignored: this node writes files
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_roc_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # One table input
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table with y and probabilities")

    # Compute and write artifacts
    lines.extend(_emit_roc_code(cfg, node_id))

    # No context outputs for this view node (by design)
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
    Entry for emitters:
      - returns (imports, body_lines) if this module can handle the node
      - returns None otherwise
    """

    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    node_lines = generate_py_body(nid, npath, in_ports)

    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
