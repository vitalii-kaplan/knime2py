#!/usr/bin/env python3

"""Scatter Plot view node."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, normalize_in_ports, split_out_imports


FACTORY = "org.knime.base.views.node.scatterplot.ScatterPlotNodeFactory"


@dataclass
class ScatterPlotSettings:
    x_col: Optional[str] = None
    y_col: Optional[str] = None
    color_col: Optional[str] = None
    title: str = "Scatter Plot"
    width: int = 800
    height: int = 600
    image_format: str = "SVG"
    max_rows: int = 2500
    point_size: int = 15
    show_legend: bool = True
    axis_extent_method: str = "AUTO"
    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = 0.0
    y_max: float = 100.0
    x_label: Optional[str] = None
    y_label: Optional[str] = None


def _bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def _float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value is not None else default
    except Exception:
        return default


def parse_scatter_plot_settings(node_dir: Optional[Path]) -> ScatterPlotSettings:
    if not node_dir:
        return ScatterPlotSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return ScatterPlotSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    view_el = first_el(root, ".//*[local-name()='config' and @key='view']")

    image_format = "SVG"
    width = 800
    height = 600
    if model_el is not None:
        image_format = first(model_el, "./*[local-name()='entry' and @key='imageFormat']/@value") or image_format
        width = _int(first(model_el, "./*[local-name()='entry' and @key='width']/@value"), width)
        height = _int(first(model_el, "./*[local-name()='entry' and @key='height']/@value"), height)

    if view_el is None:
        return ScatterPlotSettings(width=width, height=height, image_format=image_format)

    color_col = first(
        view_el,
        "./*[local-name()='config' and @key='colorColumnV2']"
        "/*[local-name()='entry' and @key='regularChoice']/@value",
    )
    x_label = first(view_el, "./*[local-name()='entry' and @key='customXAxisLabelV2']/@value")
    x_label_present = _bool(
        first(view_el, "./*[local-name()='entry' and @key='customXAxisLabelV2_is_present']/@value"),
        False,
    )
    y_label = first(view_el, "./*[local-name()='entry' and @key='customYAxisLabelV2']/@value")
    y_label_present = _bool(
        first(view_el, "./*[local-name()='entry' and @key='customYAxisLabelV2_is_present']/@value"),
        False,
    )
    manual_limits = first_el(view_el, "./*[local-name()='config' and @key='manualAxisLimits']")

    return ScatterPlotSettings(
        x_col=first(view_el, "./*[local-name()='entry' and @key='xAxisColumnV3']/@value") or None,
        y_col=first(view_el, "./*[local-name()='entry' and @key='yAxisColumnV3']/@value") or None,
        color_col=color_col or None,
        title=first(view_el, "./*[local-name()='entry' and @key='title']/@value") or "Scatter Plot",
        width=width,
        height=height,
        image_format=image_format,
        max_rows=_int(first(view_el, "./*[local-name()='entry' and @key='maxRows']/@value"), 2500),
        point_size=_int(first(view_el, "./*[local-name()='entry' and @key='dataPointSize']/@value"), 15),
        show_legend=_bool(first(view_el, "./*[local-name()='entry' and @key='showLegend']/@value"), True),
        axis_extent_method=(first(view_el, "./*[local-name()='entry' and @key='axisExtentMethod']/@value") or "AUTO").upper(),
        x_min=_float(first(manual_limits, "./*[local-name()='entry' and @key='xAxisManualMin']/@value") if manual_limits is not None else None, 0.0),
        x_max=_float(first(manual_limits, "./*[local-name()='entry' and @key='xAxisManualMax']/@value") if manual_limits is not None else None, 100.0),
        y_min=_float(first(manual_limits, "./*[local-name()='entry' and @key='yAxisManualMin']/@value") if manual_limits is not None else None, 0.0),
        y_max=_float(first(manual_limits, "./*[local-name()='entry' and @key='yAxisManualMax']/@value") if manual_limits is not None else None, 100.0),
        x_label=x_label if x_label_present and x_label else None,
        y_label=y_label if y_label_present and y_label else None,
    )


def generate_imports() -> List[str]:
    return [
        "import tempfile",
        "from pathlib import Path",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from matplotlib.colors import LinearSegmentedColormap",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.views.node.scatterplot.ScatterPlotNodeFactory"
)


def _emit_scatter_code(cfg: ScatterPlotSettings, node_id: str) -> List[str]:
    lines: List[str] = []
    lines.append(f"_x_col = {repr(cfg.x_col)}")
    lines.append(f"_y_col = {repr(cfg.y_col)}")
    lines.append(f"_color_col = {repr(cfg.color_col)}")
    lines.append(f"_title = {repr(cfg.title)}")
    lines.append(f"_x_label = {repr(cfg.x_label or cfg.x_col or 'x')}")
    lines.append(f"_y_label = {repr(cfg.y_label or cfg.y_col or 'y')}")
    lines.append(f"_width_in = {int(cfg.width)} / 100.0")
    lines.append(f"_height_in = {int(cfg.height)} / 100.0")
    lines.append(f"_img_fmt = {repr((cfg.image_format or 'SVG').upper())}")
    lines.append(f"_max_rows = int({int(cfg.max_rows)})")
    lines.append(f"_point_size = int({int(cfg.point_size)})")
    lines.append(f"_show_legend = {bool(cfg.show_legend)!r}")
    lines.append(f"_axis_extent_method = {repr(cfg.axis_extent_method.upper())}")
    lines.append(f"_manual_xlim = ({float(cfg.x_min)}, {float(cfg.x_max)})")
    lines.append(f"_manual_ylim = ({float(cfg.y_min)}, {float(cfg.y_max)})")
    lines.append("")
    lines.append("if _x_col is None or _x_col not in df.columns:")
    lines.append("    raise KeyError(f'Scatter Plot: x-axis column not found: {_x_col!r}')")
    lines.append("if _y_col is None or _y_col not in df.columns:")
    lines.append("    raise KeyError(f'Scatter Plot: y-axis column not found: {_y_col!r}')")
    lines.append("plot_df = df.head(_max_rows).copy() if _max_rows > 0 else df.copy()")
    lines.append("x = pd.to_numeric(plot_df[_x_col], errors='coerce')")
    lines.append("y = pd.to_numeric(plot_df[_y_col], errors='coerce')")
    lines.append("valid = x.notna() & y.notna()")
    lines.append("plot_df = plot_df.loc[valid].copy()")
    lines.append("x = x.loc[valid]")
    lines.append("y = y.loc[valid]")
    lines.append("")
    lines.append("_green_blue = LinearSegmentedColormap.from_list('knime2py_green_blue', ['#2ca25f', '#2b8cbe'])")
    lines.append("color_values = None")
    lines.append("if _color_col and _color_col in plot_df.columns:")
    lines.append("    c_num = pd.to_numeric(plot_df[_color_col], errors='coerce')")
    lines.append("    if c_num.notna().any():")
    lines.append("        color_values = c_num")
    lines.append("    else:")
    lines.append("        color_values = pd.Series(pd.factorize(plot_df[_color_col].astype('string'))[0], index=plot_df.index)")
    lines.append("")
    lines.append("fig, ax = plt.subplots(figsize=(_width_in, _height_in), dpi=100)")
    lines.append("if color_values is not None:")
    lines.append("    sc = ax.scatter(x, y, c=color_values, cmap=_green_blue, s=_point_size, alpha=0.85, edgecolors='none')")
    lines.append("    if _show_legend:")
    lines.append("        cbar = fig.colorbar(sc, ax=ax)")
    lines.append("        cbar.set_label(_color_col)")
    lines.append("else:")
    lines.append("    ax.scatter(x, y, color='#2b8cbe', s=_point_size, alpha=0.85, edgecolors='none')")
    lines.append("ax.set_title(_title)")
    lines.append("ax.set_xlabel(_x_label)")
    lines.append("ax.set_ylabel(_y_label)")
    lines.append("if _axis_extent_method == 'MANUAL':")
    lines.append("    ax.set_xlim(*_manual_xlim)")
    lines.append("    ax.set_ylim(*_manual_ylim)")
    lines.append("ax.grid(True, linewidth=0.5, alpha=0.35)")
    lines.append("fig.tight_layout()")
    lines.append("")
    lines.append("try:")
    lines.append("    out_dir = Path.cwd()")
    lines.append("    out_dir.mkdir(parents=True, exist_ok=True)")
    lines.append("except Exception:")
    lines.append("    out_dir = Path(tempfile.gettempdir()) / 'knime2py_scatter'")
    lines.append("    out_dir.mkdir(parents=True, exist_ok=True)")
    lines.append("ext = 'svg' if str(_img_fmt).upper() == 'SVG' else 'png'")
    lines.append(f"img_path = out_dir / ('scatter_{node_id}.' + ext)")
    lines.append("fig.savefig(img_path, bbox_inches='tight')")
    lines.append("plt.close(fig)")
    lines.append("print(f'[Scatter Plot] Wrote image to: {img_path}')")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_scatter_plot_settings(ndir)

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0] if pairs else ("UNKNOWN", "1")

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")
    lines.extend(_emit_scatter_code(cfg, node_id))

    for p in sorted({str(p or '1') for p in (out_ports or [])}):
        lines.append(f"context['{node_id}:{p}'] = df")
    return lines


def get_name() -> str:
    return "Scatter Plot"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src_id, str(getattr(edge, "source_port", "") or "1")) for src_id, edge in (incoming or [])]
    out_ports = [str(getattr(edge, "source_port", "") or "1") for _, edge in (outgoing or [])]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
