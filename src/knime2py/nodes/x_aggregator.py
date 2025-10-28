#!/usr/bin/env python3

####################################################################################################
#
# X-Aggregator
#
# Module-level contract:
#   - LOOP = "finish"  (used by emitter.py to recognize loop finish nodes)
# Generated code behavior (executed INSIDE the loop body opened by X-Partitioner):
#   - Reads the current fold DataFrame from context.
#   - Binds to the active loop state context['__loop__:<xpart_id>'] (k folds, current index).
#   - Resets the fold accumulator on the first fold to avoid stale data across runs.
#   - Optionally adds a Fold column to the current fold.
#   - Appends the current fold DF into an accumulator list stored in
#       context['__xagg__:<loop_id>:accum'].
#   - **Only on the last fold** (current == k - 1) publishes the concatenated result.
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *  # first, first_el, normalize_in_ports, collect_module_imports, split_out_imports, iter_entries

# Signal to emitter: this node finishes a loop
LOOP = "finish"

# KNIME factory (identity)
FACTORY = "org.knime.base.node.meta.xvalidation.AggregateOutputNodeFactory"

# --------------------------------------------------------------------------------------------------
# settings.xml → Settings
# --------------------------------------------------------------------------------------------------

@dataclass
class XAggSettings:
    prediction_col: Optional[str] = None
    target_col: Optional[str] = None
    add_fold_id: bool = False


def _bool(v: Optional[str], default: bool) -> bool:
    """
    Convert a string value to a boolean.

    Args:
        v (Optional[str]): The string value to convert.
        default (bool): The default boolean value if v is None.

    Returns:
        bool: The converted boolean value.
    """
    if v is None:
        return default
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def parse_xagg_settings(node_dir: Optional[Path]) -> XAggSettings:
    """
    Parse the settings from the settings.xml file.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        XAggSettings: An instance of XAggSettings populated with the parsed values.
    """
    if not node_dir:
        return XAggSettings()

    sp = node_dir / "settings.xml"
    if not sp.exists():
        return XAggSettings()

    root = ET.parse(str(sp), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")

    pred = first(model, ".//*[local-name()='entry' and @key='predictionColumn']/@value") if model is not None else None
    targ = first(model, ".//*[local-name()='entry' and @key='targetColumn']/@value") if model is not None else None
    add_fold = _bool(first(model, ".//*[local-name()='entry' and @key='addFoldId']/@value"), False) if model is not None else False

    return XAggSettings(
        prediction_col=(pred or None),
        target_col=(targ or None),
        add_fold_id=bool(add_fold),
    )

# --------------------------------------------------------------------------------------------------
# Code generators
# --------------------------------------------------------------------------------------------------

def generate_imports():
    """
    Generate the necessary import statements for the code.

    Returns:
        List[str]: A list of import statements.
    """
    return ["import pandas as pd", "import numpy as np"]


HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.base.node.meta.xvalidation.AggregateOutputNodeFactory"
)


def _emit_xagg_code(cfg: XAggSettings, node_id: str, out_ports: List[str]) -> List[str]:
    """
    Emit code that aggregates per-fold data and ONLY publishes outputs on the final fold.

    Args:
        cfg (XAggSettings): The settings for the X-Aggregator.
        node_id (str): The ID of the node.
        out_ports (List[str]): The output ports.

    Returns:
        List[str]: The generated code lines.
    """
    ports = [str(p or "1") for p in (out_ports or ["1"])]
    ports = list(dict.fromkeys(ports)) or ["1"]

    lines: List[str] = []
    lines.append("# Aggregate per-fold outputs; publish only on the final fold")
    lines.append(f"_pred_col = {repr(cfg.prediction_col) if cfg.prediction_col else 'None'}")
    lines.append(f"_targ_col = {repr(cfg.target_col) if cfg.target_col else 'None'}")
    lines.append(f"_add_fold = {bool(cfg.add_fold_id)}")
    lines.append("")
    lines.append("# Bind to an active loop state written by X-Partitioner")
    lines.append("_loop = None")
    lines.append("_loop_id = None")
    lines.append("# Prefer a loop that is not yet complete")
    lines.append("_loop_keys = [k for k in list(context.keys()) if isinstance(k, str) and k.startswith('__loop__:')]")
    lines.append("if _loop_keys:")
    lines.append("    # Try to pick a loop whose accumulator is not marked complete")
    lines.append("    _cands = []")
    lines.append("    for _lk in _loop_keys:")
    lines.append("        try:")
    lines.append("            _lid = _lk.split(':', 1)[1]")
    lines.append("        except Exception:")
    lines.append("            _lid = None")
    lines.append("        _lobj = context.get(_lk, None)")
    lines.append("        _is_done = bool(context.get(f'__xagg__:{_lid}:is_complete', False)) if _lid else False")
    lines.append("        if isinstance(_lobj, dict) and not _is_done:")
    lines.append("            _cands.append((_lid, _lobj))")
    lines.append("    if _cands:")
    lines.append("        # If multiple, choose the one with the greatest 'current' (most in-progress)")
    lines.append("        _lid, _lobj = sorted(_cands, key=lambda t: int(t[1].get('current', 0)), reverse=True)[0]")
    lines.append("        _loop_id, _loop = _lid, _lobj")
    lines.append("    else:")
    lines.append("        # Fallback: last loop key (lexicographically) if all are complete")
    lines.append("        _loop_keys.sort()")
    lines.append("        _loop_key = _loop_keys[-1]")
    lines.append("        _loop = context.get(_loop_key, None)")
    lines.append("        try:")
    lines.append("            _loop_id = _loop_key.split(':', 1)[1]")
    lines.append("        except Exception:")
    lines.append("            _loop_id = str(_loop_key)")
    lines.append("else:")
    lines.append("    _loop = None")
    lines.append(f"    _loop_id = str({repr(node_id)})")
    lines.append("")
    lines.append("_cur = int(_loop.get('current', 0)) if isinstance(_loop, dict) else 0")
    lines.append("_k   = int(_loop.get('k', 1))       if isinstance(_loop, dict) else 1")
    lines.append("")
    lines.append("# Reset accumulator at the beginning of the loop to prevent stale concatenation")
    lines.append("if int(_cur) == 0:")
    lines.append("    context.pop(f'__xagg__:{_loop_id}:accum', None)")
    lines.append("    context.pop(f'__xagg__:{_loop_id}:is_complete', None)")
    lines.append("")
    lines.append("# Current fold input")
    lines.append("cur_df = df.copy()")
    lines.append("if _add_fold:")
    lines.append("    cur_df = cur_df.copy()")
    lines.append("    cur_df['Fold'] = _cur")
    lines.append("")
    lines.append("# Optional: sanity-check columns (non-fatal)")
    lines.append("if _pred_col and _pred_col not in cur_df.columns:")
    lines.append("    _cand = [c for c in cur_df.columns if c.startswith('Prediction (')]")
    lines.append("    if _cand:")
    lines.append("        _pred_col = _cand[0]")
    lines.append("# target column is optional for this aggregation")
    lines.append("")
    lines.append("# Append this fold to an accumulator under the loop id")
    lines.append("_acc_key = f'__xagg__:{_loop_id}:accum'")
    lines.append("_acc = context.get(_acc_key, None)")
    lines.append("if not isinstance(_acc, list):")
    lines.append("    _acc = []")
    lines.append("_acc.append(cur_df)")
    lines.append("context[_acc_key] = _acc")
    lines.append("")
    lines.append("# Mark completion state and publish only on final fold")
    lines.append("context[f'__xagg__:{_loop_id}:is_complete'] = (len(_acc) >= max(int(_k), 1))")
    lines.append("if int(_cur) >= max(int(_k), 1) - 1:")
    lines.append("    out_df = pd.concat(_acc, ignore_index=True) if _acc else cur_df.iloc[0:0].copy()")
    for p in ports:
        lines.append(f"    context['{node_id}:{p}'] = out_df")
    lines.append("    # (end of loop) — downstream nodes will run once, outside the loop body")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python code body for the node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        List[str]: The generated code lines for the body.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_xagg_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Use the first connected input as per-fold data
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']  # per-fold input table")

    # Aggregate (and publish outputs only when the last fold is reached)
    lines.extend(_emit_xagg_code(cfg, node_id, out_ports or ["1"]))

    return lines


def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    """
    Generate the code for a Jupyter notebook cell.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The incoming ports.
        out_ports (Optional[List[str]]): The outgoing ports.

    Returns:
        str: The generated code as a string.
    """
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handle the node and return the necessary imports and body lines.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming connections.
        outgoing: The outgoing connections.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines.
    """
    explicit_imports = collect_module_imports(generate_imports)

    in_ports  = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
