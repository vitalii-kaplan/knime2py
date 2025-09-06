#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import * 

# KNIME factory for Partitioning
PARTITION_FACTORY = "org.knime.base.node.preproc.partition.PartitionNodeFactory"


def can_handle(node_type: Optional[str]) -> bool:
    return bool(node_type and node_type.endswith(PARTITION_FACTORY))


# ---------------------------------------------------------------------
# settings.xml → PartitionSettings
# ---------------------------------------------------------------------

@dataclass
class PartitionSettings:
    method: str = "RELATIVE"             # RELATIVE | ABSOLUTE
    sampling_method: str = "RANDOM"      # RANDOM | LINEAR | STRATIFIED
    fraction: float = 0.7                # for RELATIVE
    count: int = 100                     # for ABSOLUTE
    random_seed: Optional[int] = None
    class_column: Optional[str] = None   # for STRATIFIED


def _to_float(s: Optional[str], default: float) -> float:
    try:
        return float(s) if s is not None else default
    except Exception:
        return default


def _to_int(s: Optional[str], default: int) -> int:
    try:
        return int(s) if s is not None else default
    except Exception:
        return default


def parse_partition_settings(node_dir: Optional[Path]) -> PartitionSettings:
    if not node_dir:
        return PartitionSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return PartitionSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return PartitioningSettings()

    method = (first(model_el, ".//*[local-name()='entry' and @key='method']/@value") or "RELATIVE").upper().strip()
    sampling = (first(model_el, ".//*[local-name()='entry' and @key='samplingMethod']/@value") or "RANDOM").upper().strip()
    fraction = _to_float(first(model_el, ".//*[local-name()='entry' and @key='fraction']/@value"), 0.7)
    count = _to_int(first(model_el, ".//*[local-name()='entry' and @key='count']/@value"), 100)
    seed_raw = first(model_el, ".//*[local-name()='entry' and @key='random_seed']/@value")
    seed = _to_int(seed_raw, None) if seed_raw is not None else None
    class_col = first(model_el, ".//*[local-name()='entry' and @key='class_column']/@value")

    return PartitionSettings(
        method=method,
        sampling_method=sampling,
        fraction=fraction,
        count=count,
        random_seed=seed,
        class_column=class_col or None,
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    # pandas only (no sklearn dependency)
    return ["import pandas as pd"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.partition.PartitionNodeFactory"
)


def _emit_partition_code(cfg: PartitionSettings) -> List[str]:
    """
    Emit lines that create `train_df` and `test_df` from `df` using the Partitioning settings.
    Supports:
      - method: RELATIVE/ABSOLUTE
      - sampling_method: RANDOM | LINEAR | STRATIFIED
      - random_seed, class_column
    """
    lines: List[str] = []
    lines.append("# Copy not strictly required, but keeps pattern consistent")
    lines.append("out_df = df.copy()")
    lines.append(f"_seed = {repr(cfg.random_seed)}")

    method = (cfg.method or "RELATIVE").upper()
    sm = (cfg.sampling_method or "RANDOM").upper()

    if method == "RELATIVE":
        lines.append(f"_frac = {float(cfg.fraction)}")
        if sm == "LINEAR":
            # Deterministic head split
            lines.append("cut = int(round(len(df) * _frac))")
            lines.append("train_df = df.iloc[:cut]")
            lines.append("test_df = df.iloc[cut:]")
        elif sm == "STRATIFIED" and cfg.class_column:
            # Proportional per-class sampling; keep remainder in test
            col = repr(cfg.class_column)
            lines.append(f"parts = [g.sample(frac=_frac, random_state=_seed) for _, g in df.groupby({col}, dropna=False, sort=False)]")
            lines.append("train_df = pd.concat(parts, axis=0).sort_index() if parts else df.iloc[0:0]")
            lines.append("test_df = df.drop(train_df.index)")
        else:
            # RANDOM default
            lines.append("train_df = df.sample(frac=_frac, random_state=_seed)")
            lines.append("test_df = df.drop(train_df.index)")

    elif method == "ABSOLUTE":
        lines.append(f"_n = int({int(cfg.count)})")
        lines.append("_n = max(0, min(_n, len(df)))")
        if sm == "LINEAR":
            lines.append("train_df = df.iloc[:_n]")
            lines.append("test_df = df.iloc[_n:]")
        elif sm == "STRATIFIED" and cfg.class_column:
            # MVP: approximate via relative fraction; exact integer allocation is more involved
            lines.append("_frac = (float(_n) / max(len(df), 1)) if len(df) else 0.0")
            col = repr(cfg.class_column)
            lines.append(f"parts = [g.sample(frac=_frac, random_state=_seed) for _, g in df.groupby({col}, dropna=False, sort=False)]")
            lines.append("train_df = pd.concat(parts, axis=0).sort_index() if parts else df.iloc[0:0]")
            lines.append("# If slightly off due to rounding, trim to _n")
            lines.append("train_df = train_df.iloc[:_n]")
            lines.append("test_df = df.drop(train_df.index)")
        else:
            lines.append("train_df = df.sample(n=_n, random_state=_seed)")
            lines.append("test_df = df.drop(train_df.index)")

    else:
        lines.append(f"# TODO: Unsupported partition method '{method}'; passthrough to train=all, test=empty")
        lines.append("train_df = df.copy()")
        lines.append("test_df = df.iloc[0:0]")

    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_partition_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_partition_code(cfg))

    # Publish to context (two outputs: train, test). Respect provided port ids.
    ports = list(out_ports or []) or ["1", "2"]
    if len(ports) == 1:
        ports.append("2")  # ensure two ports exist

    # Map: first -> train, second -> test
    lines.append(f"context['{node_id}:{ports[0]}'] = train_df")
    lines.append(f"context['{node_id}:{ports[1]}'] = test_df")

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
    Central entry used by emitters:
      - returns (imports, body_lines) if this module can handle the node type
      - returns None otherwise
    """
    if not (ntype and can_handle(ntype)):
        return None

    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
