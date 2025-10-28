#!/usr/bin/env python3

"""
ROC Curve Generation Module.

Overview
----------------------------
This module generates ROC curves from a scored table based on settings.xml,
producing visualizations and CSV outputs for performance evaluation of models.

Runtime Behavior
----------------------------
Inputs:
- Reads a DataFrame from the context containing the ground truth and probability columns.

Outputs:
- Writes the generated ROC curve image and a CSV of ROC points to the context, with
  paths mapped to the node ID.

Key algorithms:
- Utilizes sklearn's roc_curve and auc functions to compute the false positive rate,
  true positive rate, and area under the curve for each specified probability column.

Edge Cases
----------------------------
The code handles missing or constant columns, NaN values, and class imbalance by
validating inputs and providing warnings for any missing probability columns.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries: pandas, numpy, sklearn,
matplotlib, and lxml. These dependencies are required for the generated code, not
for this module itself.

Usage
----------------------------
Typically invoked by upstream nodes that require ROC curve generation. An example
of expected context access is:
```
df = context['input_table']
```

Node Identity
----------------------------
KNIME factory id: org.knime.base.views.node.roccurve.ROCCurveNodeFactory

Configuration
----------------------------
The settings are defined in the `ROCCurveSettings` dataclass, which includes:
- truth_col: The column representing the ground truth (default: None).
- pos_label: The positive class label (default: None).
- proba_cols: List of probability columns to evaluate (default: empty list).
- title: Title for the ROC curve (default: "ROC Curve").
- x_label: X-axis label (default: "False positive rate (1 - specificity)").
- y_label: Y-axis label (default: "True positive rate (sensitivity)").
- width_px: Width of the output image in pixels (default: 800).
- height_px: Height of the output image in pixels (default: 600).
- image_format: Format of the output image (default: "PNG").
- line_size: Width of the ROC curve line (default: 2).

The `parse_roc_settings` function extracts these values from the settings.xml file
using XPath queries and provides fallbacks for missing entries.

Limitations
----------------------------
This module does not support automatic detection of probability columns or suffix
guessing; it strictly uses the columns specified in settings.xml.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.base.views.node.roccurve.ROCCurveNodeFactory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (  # project helpers
    first,
    first_el,
    iter_entries,
    normalize_in_ports,
    collect_module_imports,
    split_out_imports,
)

FACTORY = "org.knime.base.views.node.roccurve.ROCCurveNodeFactory"

# -------------------------------------------------------------------------------------
# settings.xml → ROCCurveSettings
# -------------------------------------------------------------------------------------

@dataclass
class ROCCurveSettings:
    truth_col: Optional[str] = None
    pos_label: Optional[str] = None
    proba_cols: List[str] = field(default_factory=list)
    title: str = "ROC Curve"
    x_label: str = "False positive rate (1 - specificity)"
    y_label: str = "True positive rate (sensitivity)"
    width_px: int = 800
    height_px: int = 600
    image_format: str = "PNG"
    line_size: int = 2


def _collect_numeric_name_entries(cfg: ET._Element) -> List[str]:
    """Collect values of <entry key='0' value='...'/>, <entry key='1' .../>, ..."""
    out: List[str] = []
    for k, v in iter_entries(cfg):
        if k.isdigit() and v:
            out.append(v.strip())
    return out


def parse_roc_settings(node_dir: Optional[Path]) -> ROCCurveSettings:
    """Parse the ROC settings from the settings.xml file located in the specified node directory."""
    if not node_dir:
        return ROCCurveSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return ROCCurveSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    view_el  = first_el(root, ".//*[local-name()='config' and @key='view']")

    # Canvas / image settings
    width_px  = int(first(model_el, ".//*[local-name()='entry' and @key='width']/@value")  or 800) if model_el is not None else 800
    height_px = int(first(model_el, ".//*[local-name()='entry' and @key='height']/@value") or 600) if model_el is not None else 600
    img_fmt   = (first(model_el, ".//*[local-name()='entry' and @key='imageFormat']/@value") or "PNG").strip().upper() if model_el is not None else "PNG"

    # Titles / labels
    title     = first(view_el, ".//*[local-name()='entry' and @key='title']/@value") or "ROC Curve"
    x_lab     = first(view_el, ".//*[local-name()='entry' and @key='xAxisLabel']/@value") or "False positive rate (1 - specificity)"
    y_lab     = first(view_el, ".//*[local-name()='entry' and @key='yAxisLabel']/@value") or "True positive rate (sensitivity)"
    line_size = int(first(view_el, ".//*[local-name()='entry' and @key='lineSize']/@value") or 2) if view_el is not None else 2

    # Truth column — support both variants
    truth_col = None
    if view_el is not None:
        # Newer: direct string entry
        truth_col = first(view_el, ".//*[local-name()='entry' and @key='targetColumnV3']/@value")
        if not truth_col:
            # Older: nested config with 'selected'
            truth_col = first(view_el, ".//*[local-name()='config' and @key='targetColumn']/*[local-name()='entry' and @key='selected']/@value")

    # Positive class label (same in both)
    pos_label = first(view_el, ".//*[local-name()='entry' and @key='positiveClassValue']/@value") if view_el is not None else None

    # Probability columns — support V2 and legacy
    proba_cols: List[str] = []
    if view_el is not None:
        # Newer: predictionColumnsV2 / manualFilter / manuallySelected
        pred_v2 = first_el(view_el, ".//*[local-name()='config' and @key='predictionColumnsV2']")
        if pred_v2 is not None:
            man_sel = first_el(pred_v2, ".//*[local-name()='config' and @key='manualFilter']/*[local-name()='config' and @key='manuallySelected']")
            if man_sel is not None:
                proba_cols.extend(_collect_numeric_name_entries(man_sel))

        # Older: predictionColumns / selected_Internals
        if not proba_cols:
            pred = first_el(view_el, ".//*[local-name()='config' and @key='predictionColumns']")
            if pred is not None:
                sel_int = first_el(pred, ".//*[local-name()='config' and @key='selected_Internals']")
                if sel_int is not None:
                    proba_cols.extend(_collect_numeric_name_entries(sel_int))
                # Also consider manualFilter/manuallySelected if present (older UIs sometimes put selections here too)
                man_sel_old = first_el(pred, ".//*[local-name()='config' and @key='manualFilter']/*[local-name()='config' and @key='manuallySelected']")
                if man_sel_old is not None:
                    proba_cols.extend(_collect_numeric_name_entries(man_sel_old))

    # Deduplicate while preserving order
    seen = set()
    proba_cols = [c for c in proba_cols if c and not (c in seen or seen.add(c))]

    return ROCCurveSettings(
        truth_col=truth_col or None,
        pos_label=pos_label or None,
        proba_cols=proba_cols,
        title=title,
        x_label=x_lab,
        y_label=y_lab,
        width_px=width_px,
        height_px=height_px,
        image_format=img_fmt or "PNG",
        line_size=line_size,
    )

# -------------------------------------------------------------------------------------
# Code generators
# -------------------------------------------------------------------------------------

def generate_imports():
    """Generate a list of import statements required for the ROC curve generation."""
    return [
        "import tempfile",
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
    """Emit the ROC curve generation code based on the provided settings and node ID."""
    lines: List[str] = []
    lines.append(f"_truth_col = {repr(cfg.truth_col) if cfg.truth_col else 'None'}")
    lines.append(f"_pos_label = {repr(cfg.pos_label) if cfg.pos_label else 'None'}")
    if cfg.proba_cols:
        cols = ", ".join(repr(c) for c in cfg.proba_cols)
        lines.append(f"_proba_cols = [{cols}]")
    else:
        lines.append("_proba_cols = []  # (empty) — using exactly the configured columns; none found in settings.xml")
    lines.append(f"_title = {repr(cfg.title)}")
    lines.append(f"_x_label = {repr(cfg.x_label)}")
    lines.append(f"_y_label = {repr(cfg.y_label)}")
    lines.append(f"_width_in = {cfg.width_px} / 100.0")
    lines.append(f"_height_in = {cfg.height_px} / 100.0")
    lines.append(f"_img_fmt = {repr((cfg.image_format or 'PNG').upper())}")
    lines.append(f"_line_width = int({cfg.line_size}) if {cfg.line_size} else 2")
    lines.append("")
    lines.append("# Validate inputs")
    lines.append("if _truth_col is None or _truth_col not in df.columns:")
    lines.append("    raise KeyError(f\"ROC: ground-truth column not found: {_truth_col!r}\")")
    lines.append("if _pos_label is None:")
    lines.append("    raise KeyError(\"ROC: positive class value is not configured in settings.xml\")")
    lines.append("if not _proba_cols:")
    lines.append("    raise KeyError('ROC: no probability columns configured in settings.xml.')")
    lines.append("")
    lines.append("# Keep only configured probability columns that are present in the table")
    lines.append("proba_cols = [c for c in _proba_cols if c in df.columns]")
    lines.append("if not proba_cols:")
    lines.append("    raise KeyError(f\"ROC: none of the configured probability columns are present: {_proba_cols}\")")
    lines.append("missing = [c for c in _proba_cols if c not in df.columns]")
    lines.append("if missing:")
    lines.append("    print(f\"[ROC] Warning: missing probability columns ignored: {missing}\")")
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
    lines.append("    plt.plot(grp['fpr'].values, grp['tpr'].values, linewidth=_line_width, label=f\"{m} (AUC={auc_val:.3f})\")")
    lines.append("plt.plot([0,1], [0,1], linestyle='--', linewidth=1)")
    lines.append("plt.title(_title)")
    lines.append("plt.xlabel(_x_label)")
    lines.append("plt.ylabel(_y_label)")
    lines.append("plt.xlim(0, 1); plt.ylim(0, 1)")
    lines.append("plt.legend(loc='lower right')")
    lines.append("")
    lines.append("# Robust output directory (cwd may be unavailable in some notebook kernels)")
    lines.append("try:")
    lines.append("    out_dir = Path.cwd()")
    lines.append("    out_dir.mkdir(parents=True, exist_ok=True)")
    lines.append("except Exception:")
    lines.append("    import tempfile")
    lines.append("    out_dir = Path(tempfile.gettempdir()) / 'knime2py_roc'")
    lines.append("    out_dir.mkdir(parents=True, exist_ok=True)")
    lines.append("ext = 'svg' if str(_img_fmt).upper() == 'SVG' else 'png'")
    lines.append(f"img_path = out_dir / ('roc_{node_id}.' + ext)")
    lines.append(f"csv_path = out_dir / ('roc_table_{node_id}.csv')")
    lines.append("plt.savefig(img_path, bbox_inches='tight')")
    lines.append("plt.close()")
    lines.append("points_df.to_csv(csv_path, index=False)")
    lines.append("print(f'[ROC] Wrote image to: {img_path}')")
    lines.append("print(f'[ROC] Wrote points to: {csv_path}')")
    return lines


def generate_imports():
    """Generate a list of import statements required for the ROC curve generation."""
    return [
        "import tempfile",
        "from pathlib import Path",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from sklearn.metrics import roc_curve, auc",
    ]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,  # ignored
) -> List[str]:
    """Generate the Python body for the ROC curve generation based on the node ID and input ports."""
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_roc_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table with y and probabilities")

    lines.extend(_emit_roc_code(cfg, node_id))
    return lines


def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    """Generate the code for a Jupyter notebook cell for the ROC curve generation."""
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """Handle the node processing by generating the necessary imports and body code."""
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    node_lines = generate_py_body(nid, npath, in_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
