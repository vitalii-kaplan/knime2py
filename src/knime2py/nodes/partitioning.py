#!/usr/bin/env python3

####################################################################################################
#
# Partitioning
#
# Splits an input table into train/test using settings.xml. Supports RELATIVE (fraction) or ABSOLUTE
# (row count) sizing with RANDOM, LINEAR (no shuffle), or STRATIFIED sampling. Emits two outputs:
# train_df and test_df, written to the node's context.
#
# - Implementation: sklearn.model_selection.train_test_split; seed honored when provided.
# - STRATIFIED: uses class_column; NaN treated as a separate class; falls back to non-stratified if
#   stratification is infeasible (e.g., tiny classes).
# - RELATIVE: fraction is clamped to [0,1]. ABSOLUTE: train_size is an integer bounded by len(df).
# 
####################################################################################################


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, normalize_in_ports, collect_module_imports, split_out_imports, iter_entries, etc.

# KNIME factory for Partitioning
FACTORY = "org.knime.base.node.preproc.partition.PartitionNodeFactory"

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
    """
    Convert a string to a float, returning a default value if conversion fails.

    Args:
        s (Optional[str]): The string to convert.
        default (float): The default value to return on failure.

    Returns:
        float: The converted float value or the default.
    """
    try:
        return float(s) if s is not None else default
    except Exception:
        return default


def _to_int(s: Optional[str], default: Optional[int]) -> Optional[int]:
    """
    Convert a string to an integer, returning a default value if conversion fails.

    Args:
        s (Optional[str]): The string to convert.
        default (Optional[int]): The default value to return on failure.

    Returns:
        Optional[int]: The converted integer value or the default.
    """
    try:
        return int(s) if s is not None else default
    except Exception:
        return default


def parse_partition_settings(node_dir: Optional[Path]) -> PartitionSettings:
    """
    Parse the partition settings from the settings.xml file.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        PartitionSettings: The parsed partition settings.
    """
    if not node_dir:
        return PartitionSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return PartitionSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")
    if model_el is None:
        return PartitionSettings()

    method = (first(model_el, ".//*[local-name()='entry' and @key='method']/@value") or "RELATIVE").upper().strip()
    sampling = (first(model_el, ".//*[local-name()='entry' and @key='samplingMethod']/@value") or "RANDOM").upper().strip()
    fraction = _to_float(first(model_el, ".//*[local-name()='entry' and @key='fraction']/@value"), 0.7)
    count = _to_int(first(model_el, ".//*[local-name()='entry' and @key='count']/@value"), 100) or 100
    seed = _to_int(first(model_el, ".//*[local-name()='entry' and @key='random_seed']/@value"), None)
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
    """
    Generate the necessary import statements for the partitioning code.

    Returns:
        List[str]: A list of import statements.
    """
    # now depends on scikit-learn
    return [
        "import pandas as pd",
        "from sklearn.model_selection import train_test_split",
    ]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.preproc.partition.PartitionNodeFactory"
)


def _emit_partition_code(cfg: PartitionSettings) -> List[str]:
    """
    Emit lines that create `train_df` and `test_df` from `df` using sklearn.model_selection.train_test_split.
    Supports:
      - method: RELATIVE/ABSOLUTE
      - sampling_method: RANDOM | LINEAR | STRATIFIED
      - random_seed, class_column (for STRATIFIED)

    Args:
        cfg (PartitionSettings): The partition settings configuration.

    Returns:
        List[str]: The emitted lines of code for partitioning.
    """
    lines: List[str] = []
    lines.append("# Copy not strictly required, but keeps pattern consistent")
    lines.append("out_df = df.copy()")
    lines.append(f"_seed = {repr(cfg.random_seed)}")

    method = (cfg.method or "RELATIVE").upper()
    sm = (cfg.sampling_method or "RANDOM").upper()

    if method == "RELATIVE":
        # clamp fraction into [0,1]
        frac_val = max(0.0, min(1.0, float(cfg.fraction)))
        lines.append(f"_frac = {frac_val}")

        if sm == "LINEAR":
            # deterministic split by order
            lines.append("train_df, test_df = train_test_split(df, train_size=_frac, shuffle=False, random_state=_seed)")
        elif sm == "STRATIFIED" and cfg.class_column:
            col = repr(cfg.class_column)
            # emulate KNIME dropna=False by treating NaN as its own class
            lines.append(f"_y = df[{col}].astype('object').where(pd.notna(df[{col}]), '__NA__')")
            lines.append("try:")
            lines.append("    train_df, test_df = train_test_split(df, train_size=_frac, random_state=_seed, stratify=_y)")
            lines.append("except Exception:")
            lines.append("    # Fallback if stratification fails (e.g., tiny classes)")
            lines.append("    train_df, test_df = train_test_split(df, train_size=_frac, random_state=_seed)")
        else:
            # RANDOM default
            lines.append("train_df, test_df = train_test_split(df, train_size=_frac, random_state=_seed)")

    elif method == "ABSOLUTE":
        lines.append(f"_n = int({int(cfg.count)})")
        lines.append("_n = max(0, min(_n, len(df)))")
        if sm == "LINEAR":
            lines.append("train_df, test_df = train_test_split(df, train_size=_n, shuffle=False, random_state=_seed)")
        elif sm == "STRATIFIED" and cfg.class_column:
            col = repr(cfg.class_column)
            lines.append(f"_y = df[{col}].astype('object').where(pd.notna(df[{col}]), '__NA__')")
            lines.append("try:")
            lines.append("    train_df, test_df = train_test_split(df, train_size=_n, random_state=_seed, stratify=_y)")
            lines.append("except Exception:")
            lines.append("    train_df, test_df = train_test_split(df, train_size=_n, random_state=_seed)")
        else:
            lines.append("train_df, test_df = train_test_split(df, train_size=_n, random_state=_seed)")
    else:
        lines.append(f"# TODO: Unsupported partition method '{method}'; passthrough to train=all, test=empty")
        lines.append("train_df = df.copy()")   # all rows
        lines.append("test_df = df.iloc[0:0]") # empty

    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python code body for the partitioning node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The list of incoming ports.
        out_ports (Optional[List[str]]): The list of outgoing ports.

    Returns:
        List[str]: The generated lines of Python code.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_partition_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # input table")

    lines.extend(_emit_partition_code(cfg))

    # Publish to context (two outputs: train, test). Ensure two unique ports exist.
    ports = [str(p or "1") for p in (out_ports or ["1", "2"])]
    if len(ports) == 1:
        ports.append("2")
    # de-dupe while preserving order a bit; if it collapses to one, add a second
    ports = list(dict.fromkeys(ports))
    if len(ports) == 1:
        ports.append("2" if ports[0] != "2" else "1")

    # Sort ports so the *smaller* port id receives train_df
    def _port_key(p: str):
        # numeric ports sort numerically; non-numeric sort after numerics, lexicographically
        return (0, int(p)) if p.isdigit() else (1, p)

    p_train, p_test = sorted(ports, key=_port_key)[:2]

    lines.append(f"context['{node_id}:{p_train}'] = train_df")
    lines.append(f"context['{node_id}:{p_test}'] = test_df")

    return lines



def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    """
    Generate the code for a Jupyter notebook cell for the partitioning node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The list of incoming ports.
        out_ports (Optional[List[str]]): The list of outgoing ports.

    Returns:
        str: The generated code as a string.
    """
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Central entry used by emitters to handle the node type.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming connections.
        outgoing: The outgoing connections.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines if this module can handle the node type,
        or None otherwise.
    """

    explicit_imports = collect_module_imports(generate_imports)

    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1", "2"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
