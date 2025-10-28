#!/usr/bin/env python3

####################################################################################################
#
# CSV Writer
# 
# Writes a pandas DataFrame to CSV using options parsed from settings.xml.#
# Resolves LOCAL/RELATIVE (knime.workflow) paths and maps KNIME writer options to pandas.to_csv.
#
# pandas>=1.5 recommended for consistent NA/nullable dtype handling.
# Path resolution supports LOCAL absolute paths and RELATIVE knime.workflow; other FS types are not yet handled.
# Directory creation is not automatic; ensure out_path.parent exists before writing.
# Line terminator / quoting mode / doublequote / escapechar are not explicitly mapped unless present; pandas defaults apply.
# File is overwritten by default; KNIME “append/overwrite” style flags are not implemented here.
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import * 

FACTORY = "org.knime.base.node.io.filehandling.csv.writer.CSVWriter2NodeFactory"

@dataclass
class CSVWriterSettings:
    path: Optional[str] = None
    sep: Optional[str] = ","
    quotechar: Optional[str] = '"'
    header: Optional[bool] = True
    encoding: Optional[str] = "utf-8"
    na_rep: Optional[str] = None   # representation for NaN, e.g. "" or "null"
    include_index: bool = False    # pandas index to file?


# ----------------------------
# Read settings.xml → CSVWriterSettings
# ----------------------------

def parse_csv_writer_settings(node_dir: Optional[Path]) -> CSVWriterSettings:
    """
    Read <node_dir>/settings.xml and extract CSV Writer options.
    Path resolution:
      - LOCAL => absolute path from settings
      - RELATIVE + knime.workflow => path relative to the workflow root (node_dir)
    
    Args:
        node_dir (Optional[Path]): The directory of the node containing settings.xml.

    Returns:
        CSVWriterSettings: An object containing the parsed settings for the CSV writer.
    """
    if not node_dir:
        return CSVWriterSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return CSVWriterSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    # Path: prefer robust resolver (handles LOCAL vs RELATIVE/knime.workflow),
    # fallback to legacy extractor if anything goes wrong.
    out_path: Optional[str]
    try:
        resolved = resolve_reader_path(root, node_dir)  # shared helper moved to node_utils.py
        out_path = str(resolved) if resolved is not None else None
    except Exception:
        out_path = None
    if out_path is None:
        out_path = extract_csv_path(root)

    # Other writer options
    sep = extract_csv_sep(root) or ","
    quotechar = extract_csv_quotechar(root) or '"'
    header = extract_csv_header_writer(root)
    if header is None:
        header = True

    enc = extract_csv_encoding(root) or "utf-8"
    na_rep = extract_csv_na_rep(root)           # keep '' if present
    include_index = extract_csv_include_index(root)
    if include_index is None:
        include_index = False

    return CSVWriterSettings(
        path=out_path,
        sep=sep,
        quotechar=quotechar,
        header=header,
        encoding=enc,
        na_rep=na_rep,
        include_index=include_index,
    )



# ----------------------------
# Code generators
# ----------------------------

def generate_imports():
    """
    Generate a list of import statements required for the CSV writer.

    Returns:
        List[str]: A list of import statements.
    """
    return ["from pathlib import Path", "import pandas as pd"]

def _fmt_kw(key: str, val) -> Optional[str]:
    """
    Format a keyword argument for the to_csv function.

    Args:
        key (str): The name of the keyword argument.
        val: The value of the keyword argument.

    Returns:
        Optional[str]: A formatted string in the form 'key=value' or None if the value should be skipped.
    """
    if isinstance(val, bool):
        return f"{key}={'True' if val else 'False'}"
    if val is None:
        # Keep explicit None only for na_rep; skip for others if you prefer.
        return f"{key}=None" if key == "na_rep" else None
    return f"{key}={repr(val)}"


def generate_py_body(node_id: str, node_dir: Optional[str], in_ports: List[object]) -> List[str]:
    """
    Generate the body of the Python function for the CSV Writer node.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The input ports for the node.

    Returns:
        List[str]: A list of lines representing the body of the function.
    """
    ndir = Path(node_dir) if node_dir else None
    settings = parse_csv_writer_settings(ndir) if ndir else CSVWriterSettings()

    lines: List[str] = []
    lines.append("# https://hub.knime.com/knime/extensions/org.knime.features.base/latest/" + FACTORY)

    # Pull input dataframe from context (CSV Writer has a single table input)
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']")

    # Output path
    if settings.path:
        lines.append(f"out_path = Path(r\"{settings.path}\")")
    else:
        lines.append("# WARNING: output CSV path not found in settings.xml. Please set manually:")
        lines.append("out_path = Path('path/to/output.csv')")

    # Build to_csv kwargs
    kw_parts = [
        _fmt_kw("sep", settings.sep or ","),
        _fmt_kw("quotechar", settings.quotechar or '"'),
        _fmt_kw("header", bool(settings.header)),
        _fmt_kw("encoding", settings.encoding or "utf-8"),
        _fmt_kw("na_rep", settings.na_rep),          # may be '' (empty string)
        _fmt_kw("index", bool(settings.include_index)),
    ]
    kw_str = ", ".join(p for p in kw_parts if p is not None)

    lines.append(f"df.to_csv(out_path, {kw_str})")
    return lines


def generate_ipynb_code(node_id: str, node_dir: Optional[str], in_ports: List[object]) -> str:
    """
    Generate the complete code for the CSV Writer node in IPython notebook format.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The input ports for the node.

    Returns:
        str: The generated code as a string.
    """
    body = generate_py_body(node_id, node_dir, in_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handle the processing of the CSV Writer node, generating the necessary imports and body code.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming connections to the node.
        outgoing: The outgoing connections from the node.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing a list of import statements and the body code.
    """
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in incoming]
    node_lines = generate_py_body(nid, npath, in_ports)

    found, body = split_out_imports(node_lines)
    explicit = collect_module_imports(generate_imports)
    imports = sorted(set(found).union(explicit))
    return imports, body
