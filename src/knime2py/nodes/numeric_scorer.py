#!/usr/bin/env python3

"""Numeric Scorer node."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, normalize_in_ports, split_out_imports


FACTORY = "org.knime.base.node.mine.scorer.numeric2.NumericScorer2NodeFactory"


@dataclass
class NumericScorerSettings:
    reference_col: str = "target"
    predicted_col: str = "prediction"
    override_output_name: bool = False
    output_col: Optional[str] = None
    number_of_predictors: int = 0


def _bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def parse_numeric_scorer_settings(node_dir: Optional[Path]) -> NumericScorerSettings:
    if not node_dir:
        return NumericScorerSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return NumericScorerSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return NumericScorerSettings()

    reference_col = (
        first(
            model_el,
            "./*[local-name()='config' and @key='reference']"
            "/*[local-name()='entry' and @key='columnName']/@value",
        )
        or "target"
    )
    predicted_col = (
        first(
            model_el,
            "./*[local-name()='config' and @key='predicted']"
            "/*[local-name()='entry' and @key='columnName']/@value",
        )
        or "prediction"
    )
    override_output_name = _bool(
        first(model_el, "./*[local-name()='entry' and @key='override default output name']/@value"),
        False,
    )
    output_col = first(model_el, "./*[local-name()='entry' and @key='output column']/@value") or None
    number_of_predictors = _int(
        first(model_el, "./*[local-name()='entry' and @key='number_of_predictors']/@value"),
        0,
    )

    return NumericScorerSettings(
        reference_col=reference_col,
        predicted_col=predicted_col,
        override_output_name=override_output_name,
        output_col=output_col,
        number_of_predictors=number_of_predictors,
    )


def generate_imports() -> List[str]:
    return ["import pandas as pd", "import numpy as np"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.mine.scorer.numeric2.NumericScorer2NodeFactory"
)


def _emit_score_code(cfg: NumericScorerSettings) -> List[str]:
    lines: List[str] = []
    lines.append(f"_reference_col = {repr(cfg.reference_col)}")
    lines.append(f"_predicted_col = {repr(cfg.predicted_col)}")
    lines.append("missing = [c for c in [_reference_col, _predicted_col] if c not in df.columns]")
    lines.append("if missing:")
    lines.append("    raise KeyError(f'Numeric Scorer: missing required column(s): {missing}')")
    lines.append("pair = df[[_reference_col, _predicted_col]].copy()")
    lines.append("pair[_reference_col] = pd.to_numeric(pair[_reference_col], errors='coerce')")
    lines.append("pair[_predicted_col] = pd.to_numeric(pair[_predicted_col], errors='coerce')")
    lines.append("pair = pair.dropna(subset=[_reference_col, _predicted_col])")
    lines.append("y_true = pair[_reference_col].to_numpy(dtype=float)")
    lines.append("y_pred = pair[_predicted_col].to_numpy(dtype=float)")
    lines.append("err = y_true - y_pred")
    lines.append("n = int(len(y_true))")
    lines.append("if n == 0:")
    lines.append("    r2 = mae = mse = rmse = mean_signed = mape = adjusted_r2 = float('nan')")
    lines.append("else:")
    lines.append("    sse = float(np.sum(err ** 2))")
    lines.append("    centered = y_true - float(np.mean(y_true))")
    lines.append("    sst = float(np.sum(centered ** 2))")
    lines.append("    r2 = float(1.0 - sse / sst) if sst != 0 else float('nan')")
    lines.append("    mae = float(np.mean(np.abs(err)))")
    lines.append("    mse = float(np.mean(err ** 2))")
    lines.append("    rmse = float(np.sqrt(mse))")
    lines.append("    mean_signed = float(np.mean(err))")
    lines.append("    if np.any(y_true == 0):")
    lines.append("        mape = 'NaN'")
    lines.append("    else:")
    lines.append("        mape = float(np.mean(np.abs(err / y_true)))")
    lines.append(f"    p = int({int(cfg.number_of_predictors)})")
    lines.append("    denom = n - p - 1")
    lines.append("    adjusted_r2 = float(1.0 - (1.0 - r2) * (n - 1) / denom) if denom > 0 and pd.notna(r2) else float('nan')")
    output_name = cfg.output_col if cfg.override_output_name and cfg.output_col else cfg.predicted_col
    lines.append(f"_score_col = {repr(output_name)}")
    lines.append("out_df = pd.DataFrame({_score_col: [r2, mae, mse, rmse, mean_signed, mape, adjusted_r2]})")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_numeric_scorer_settings(ndir)

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0] if pairs else ("UNKNOWN", "1")

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")
    lines.extend(_emit_score_code(cfg))

    for p in sorted({str(p or '1') for p in (out_ports or ['1'])}):
        lines.append(f"context['{node_id}:{p}'] = out_df")
    return lines


def get_name() -> str:
    return "Numeric Scorer"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src_id, str(getattr(edge, "source_port", "") or "1")) for src_id, edge in (incoming or [])]
    out_ports = [str(getattr(edge, "source_port", "") or "1") for _, edge in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
