#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (
    first, bool_from_value, normalize_delim, normalize_char, looks_like_path,
    normalize_in_ports,
    extract_csv_path, extract_csv_sep, extract_csv_quotechar,
    extract_csv_encoding, extract_csv_header_writer,
)


CSV_WRITER_FACTORY = "org.knime.base.node.io.filehandling.csv.writer.CSVWriter2NodeFactory"


def can_handle(node_type: Optional[str]) -> bool:
    """Return True if this generator supports the node factory."""
    if not node_type:
        return False
    return node_type.endswith(".CSVWriter2NodeFactory")


@dataclass
class CSVWriterSettings:
    path: Optional[str] = None
    sep: Optional[str] = ","
    quotechar: Optional[str] = '"'
    header: Optional[bool] = True
    encoding: Optional[str] = "utf-8"
    na_rep: Optional[str] = None         # representation for NaN, e.g. "" or "null"
    include_index: bool = False          # pandas index to file?


# ----------------------------
# Read settings.xml → CSVWriterSettings
# ----------------------------

def parse_csv_writer_settings(node_dir: Optional[Path]) -> CSVWriterSettings:
    if not node_dir:
        return CSVWriterSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return CSVWriterSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    path = extract_csv_path(root)
    sep = extract_csv_sep(root) or ","
    quotechar = extract_csv_quotechar(root) or '"'
    header = extract_csv_header_writer(root)
    if header is None:
        header = True
    enc = extract_csv_encoding(root) or "utf-8"

    na_rep = first(
        root,
        ".//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'missing')"
        " and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'representation')]/@value"
    )

    include_index_raw = first(
        root,
        ".//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'includeindex')]/@value"
    )
    include_index = bool_from_value(include_index_raw)
    if include_index is None:
        include_index = False

    return CSVWriterSettings(
        path=path,
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

def generate_py_body(node_id: str, node_dir: Optional[str], in_ports: List[object]) -> List[str]:
    """
    Return the *body* lines to place inside the function for this CSV Writer node
    in the .py workbook. Accepts input ports as either [('1393','1')] or ['1393:1'].
    """
    ndir = Path(node_dir) if node_dir else None
    settings = parse_csv_writer_settings(ndir) if ndir else CSVWriterSettings()

    lines: List[str] = []
    # Link to hub doc
    lines.append("# https://hub.knime.com/knime/extensions/org.knime.features.base/latest/" + CSV_WRITER_FACTORY)
    lines.append("from pathlib import Path")
    lines.append("import pandas as pd")

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

    # Build to_csv kwargs (precompute reprs to avoid f-string escape issues)
    sep_repr = repr(settings.sep) if settings.sep is not None else repr(",")
    quote_repr = repr(settings.quotechar) if settings.quotechar is not None else repr('"')
    enc_repr = repr(settings.encoding) if settings.encoding else repr("utf-8")
    na_rep_repr = repr(settings.na_rep) if settings.na_rep is not None else "None"
    index_bool = "True" if settings.include_index else "False"
    header_bool = "True" if settings.header else "False"

    # Emit to_csv
    line = (
        "df.to_csv("
        "out_path, "
        f"sep={sep_repr}, "
        f"quotechar={quote_repr}, "
        f"header={header_bool}, "
        f"encoding={enc_repr}, "
        f"na_rep={na_rep_repr}, "
        f"index={index_bool}"
        ")"
    )
    lines.append(line)

    return lines


def generate_ipynb_code(node_id: str, node_dir: Optional[str], in_ports: List[object]) -> str:
    """
    Return the code cell text for the notebook workbook (single string).
    """
    body = generate_py_body(node_id, node_dir, in_ports)
    return "\n".join(body) + "\n"
