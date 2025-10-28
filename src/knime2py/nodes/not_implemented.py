#!/usr/bin/env python3
from __future__ import annotations

import keyword
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (
    first,
    first_el,
    iter_entries,
    collect_module_imports,
    split_out_imports,
)

FACTORY = ""

# Fallback handler for any node type that we don't implement yet.
# IMPORTANT: ensure this module is appended LAST in get_handlers().


PRIORITY = 1_000_000

# ----------------------------
# Helpers
# ----------------------------

_py_ident_rx = re.compile(r"[^0-9a-zA-Z_]+")


def _sanitize_name(key: str, used: set[str]) -> str:
    """
    Turn an arbitrary KNIME entry key into a safe Python identifier.
    Deduplicate by suffixing _2, _3, ... as needed.
    
    Args:
        key (str): The KNIME entry key to sanitize.
        used (set[str]): A set of already used names to avoid duplicates.

    Returns:
        str: A sanitized Python identifier.
    """
    base = key.strip() or "param"
    base = _py_ident_rx.sub("_", base).strip("_")
    if not base:
        base = "param"
    # identifiers can't start with a digit
    if base[0].isdigit():
        base = f"p_{base}"
    # avoid keywords & builtins-looking names
    if keyword.iskeyword(base):
        base = f"{base}_"
    name = base
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def _coerce_literal(v: Optional[str]):
    """
    Best-effort literal coercion:
      - "true"/"false"/"1"/"0"/"yes"/"no" → bool
      - ints, floats → numeric
      - else → repr(string)
    
    Args:
        v (Optional[str]): The string value to coerce.

    Returns:
        str: The coerced literal as a string.
    """
    if v is None:
        return "None"
    s = v.strip()
    if s == "":
        return "''"  # preserve explicit empty string
    low = s.lower()
    if low in {"true", "1", "yes", "y"}:
        return "True"
    if low in {"false", "0", "no", "n"}:
        return "False"
    # int?
    try:
        i = int(s)
        return str(i)
    except Exception:
        pass
    # float?
    try:
        f = float(s)
        # Keep as plain float, but ensure typical repr
        return repr(f)
    except Exception:
        pass
    # fallback to string
    return repr(s)


def _collect_model_params(model_el: ET._Element) -> List[Tuple[str, str]]:
    """
    Return a list of (py_name, py_literal) extracted from all <entry> under the model config.
    
    Args:
        model_el (ET._Element): The XML element representing the model configuration.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing Python names and their corresponding literals.
    """
    used: set[str] = set()
    out: List[Tuple[str, str]] = []
    for k, v in iter_entries(model_el):
        if not k:
            # make a lightweight key based on path order
            k = "param"
        py_name = _sanitize_name(k, used)
        py_lit = _coerce_literal(v)
        out.append((py_name, py_lit))
    return out


# ----------------------------
# Code generators
# ----------------------------

def generate_imports() -> List[str]:
    """
    Generate a list of imports required for the node.
    
    Returns:
        List[str]: A list of import statements.
    """
    # No mandatory imports for the stub; keep it minimal.
    return []


def _emit_params_block(node_dir: Optional[Path]) -> List[str]:
    """
    Build the lines that declare variables for all <entry> under <config key="model">.
    
    Args:
        node_dir (Optional[Path]): The directory containing the node's settings.

    Returns:
        List[str]: A list of lines declaring the parameters.
    """
    lines: List[str] = []

    if not node_dir:
        lines.append("# settings.xml not found; nothing to extract")
        return lines

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        lines.append("# settings.xml not found; nothing to extract")
        return lines

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model_el = first_el(root, ".//*[local-name()='config' and @key='model']")

    # Optional: include a Hub/factory hint if present
    fac = first(root, ".//*[local-name()='entry' and @key='factory']/@value")
    if fac:
        lines.append(f"# factory: {fac}")

    if model_el is None:
        lines.append("# <config key='model'> is not present in settings.xml")
        return lines

    lines.append("# Parameters discovered in <config key='model'>:")
    params = _collect_model_params(model_el)
    if not params:
        lines.append("# (no <entry> items found under model)")
        return lines

    for name, lit in params:
        lines.append(f"{name} = {lit}")
    return lines


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate the Python body for the node, including parameters from settings.xml.
    
    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory containing the node's settings.
        in_ports (List[object]): The list of incoming ports.
        out_ports (Optional[List[str]]): The list of outgoing ports.

    Returns:
        List[str]: A list of lines representing the Python body of the node.
    """
    ndir = Path(node_dir) if node_dir else None

    lines: List[str] = []
    lines.append("# This node type is not implemented in knime2py yet.")
    lines.append("# Below are variables initialized from settings.xml.")
    lines.extend(_emit_params_block(ndir))
    lines.append("")
    lines.append("# TODO: implement this node’s logic, reading inputs from `context` if needed.")
    lines.append("pass")
    return lines


def generate_ipynb_code(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[object],
    out_ports: Optional[List[str]] = None,
) -> str:
    """
    Generate the code for the Jupyter notebook cell corresponding to the node.
    
    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory containing the node's settings.
        in_ports (List[object]): The list of incoming ports.
        out_ports (Optional[List[str]]): The list of outgoing ports.

    Returns:
        str: The generated code as a string.
    """
    body = generate_py_body(node_id, node_dir, in_ports, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handle the node processing, returning imports and body lines.
    
    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path to the node.
        incoming: The incoming ports.
        outgoing: The outgoing ports.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the list of imports and the body lines.
    """
    explicit_imports = collect_module_imports(generate_imports)

    node_lines = generate_py_body(nid, npath, incoming, outgoing)
    found_imports, body = split_out_imports(node_lines)

    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
