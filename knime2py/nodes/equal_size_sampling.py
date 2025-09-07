#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  

# KNIME factory for Equal Size Sampling
EQUAL_SIZE_SAMPLING_FACTORY = "org.knime.base.node.preproc.equalsizesampling.EqualSizeSamplingNodeFactory"

def can_handle(node_type: Optional[str]) -> bool:
    return bool(node_type and node_type.endswith(EQUAL_SIZE_SAMPLING_FACTORY))


# ---------------------------------------------------------------------
# settings.xml → EqualSizeSamplingSettings
# ---------------------------------------------------------------------

@dataclass
class EqualSizeSamplingSettings:
    class_col: Optional[str] = None
    seed: int = 1
    method: str = "Exact"   # KNIME exposes "Exact" vs "Approximate"; we implement Exact (downsample to min)

def parse_equal_size_sampling_settings(node_dir: Optional[Path]) -> EqualSizeSamplingSettings:
    """
    Parse Equal Size Sampling settings:
      - classColumn (target/stratify column)
      - seed
      - samplingMethod ("Exact" | "Approximate")
    """
    if not node_dir:
        return EqualSizeSamplingSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return EqualSizeSamplingSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model is None:
        return EqualSizeSamplingSettings()

    class_col = first(model, ".//*[local-name()='entry' and @key='classColumn']/@value")
    method = (first(model, ".//*[local-name()='entry' and @key='samplingMethod']/@value") or "Exact").strip()
    seed_raw = first(model, ".//*[local-name()='entry' and @key='seed']/@value")
    try:
        seed = int(seed_raw) if seed_raw is not None else 1
    except Exception:
        seed = 1

    return EqualSizeSamplingSettings(class_col=class_col or None, seed=seed, method=method or "Exact")


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    return ["import pandas as pd"]

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.equalsizesampling.EqualSizeSamplingNodeFactory"
)

def _emit_equal_size_code(cfg: EqualSizeSamplingSettings) -> List[str]:
    """
    Emit lines that create `out_df` by downsampling each class to the size of the smallest class.
    Only the 'Exact' method is implemented (KNIME's typical equal-size behavior).
    """
    lines: List[str] = []
    lines.append("out_df = df.copy()")
    if not cfg.class_col:
        lines.append("# No class column configured; passthrough.")
        return lines

    lines.append(f"_class = {repr(cfg.class_col)}")
    lines.append(f"_seed = {cfg.seed}")

    # Exact equal-size downsampling to the minimum class size
    lines.append("if df.empty:")
    lines.append("    out_df = df.iloc[0:0]")
    lines.append("else:")
    lines.append("    min_count = min(len(g) for _, g in df.groupby(_class, dropna=False, sort=False))")
    lines.append("    parts = [g.sample(n=min_count, random_state=_seed) for _, g in df.groupby(_class, dropna=False, sort=False)]")
    lines.append("    out_df = pd.concat(parts, axis=0).sort_index() if parts else df.iloc[0:0]")

    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_equal_size_sampling_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_equal_size_code(cfg))

    # Publish to context (default port '1')
    ports = out_ports or ["1"]
    for p in sorted({(p or '1') for p in ports}):
        lines.append(f"context['{node_id}:{p}'] = out_df")

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
    Entry used by emitters:
      - returns (imports, body_lines) if this module can handle the node type
      - returns None otherwise
    """
    if not (ntype and can_handle(ntype)):
        return None

    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
