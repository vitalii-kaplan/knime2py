#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER  # project helper (ok)
from .node_utils import (
    first,
    all_values,
    normalize_delim,
    normalize_char,
    looks_like_path,
    bool_from_value,
    context_assignment_lines,
)

CSV_FACTORY = "org.knime.base.node.io.filehandling.csv.reader.CSVTableReaderNodeFactory"


def can_handle(node_type: Optional[str]) -> bool:
    """Return True if this generator supports the node factory."""
    if not node_type:
        return False
    return node_type.endswith(".CSVTableReaderNodeFactory")


@dataclass
class CSVReaderSettings:
    path: Optional[str] = None
    sep: Optional[str] = None
    quotechar: Optional[str] = None
    escapechar: Optional[str] = None
    header: Optional[bool] = None
    encoding: Optional[str] = None


# ----------------------------
# Settings.xml → CSVReaderSettings
# ----------------------------

def parse_csv_reader_settings(node_dir: Path) -> CSVReaderSettings:
    """
    Read <node_dir>/settings.xml and extract the csv file path and common options.
    Heuristic but robust across KNIME 5.x file-handling variants.
    """
    settings = node_dir / "settings.xml"
    if not settings.exists():
        return CSVReaderSettings()

    root = ET.parse(str(settings), parser=XML_PARSER).getroot()

    # 1) File path candidates
    path_candidates = all_values(
        root,
        "(.//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'path')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'url')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'file')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'location')]/@value)"
    )
    file_path = next((p for p in path_candidates if looks_like_path(p)), None)

    # 2) Delimiter
    delim_raw = first(
        root,
        "(.//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'delim')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'separator')]/@value)"
    )
    sep = normalize_delim(delim_raw) or ","

    # 3) Quote
    quote_raw = first(
        root,
        ".//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'quote')]/@value"
    )
    quotechar = normalize_char(quote_raw) or '"'

    # 4) Escape
    esc_raw = first(
        root,
        ".//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'escape')]/@value"
    )
    escapechar = normalize_char(esc_raw)

    # 5) Header
    header_raw = first(
        root,
        "(.//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'column')"
        " and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'header')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'hasheader')]/@value"
        " | .//*[local-name()='entry' and @key='header']/@value)"
    )
    header = bool_from_value(header_raw)
    if header is None:
        header = True

    # 6) Encoding
    enc = first(
        root,
        "(.//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'charset')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,"
        " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'encoding')]/@value)"
    ) or "utf-8"

    return CSVReaderSettings(
        path=file_path,
        sep=sep,
        quotechar=quotechar,
        escapechar=escapechar,
        header=header,
        encoding=enc,
    )


# ----------------------------
# Code generators
# ----------------------------

def generate_py_body(node_id: str, node_dir: Optional[str], out_ports: List[str]) -> List[str]:
    """
    Return the *body* lines to place inside the function for this node in the .py workbook.
    (Emitters handle 'def node_...():' wrapper and indentation.)
    """
    ndir = Path(node_dir) if node_dir else None
    settings = parse_csv_reader_settings(ndir) if ndir else CSVReaderSettings()

    lines: List[str] = []
    # link to hub doc
    lines.append("# https://hub.knime.com/knime/extensions/org.knime.features.base/latest/" + CSV_FACTORY)
    lines.append("from pathlib import Path")
    lines.append("import pandas as pd")

    # Path
    if settings.path:
        lines.append(f"csv_path = Path(r\"{settings.path}\")")
    else:
        lines.append("# WARNING: CSV path not found in settings.xml. Please set manually:")
        lines.append("csv_path = Path('path/to/file.csv')")

    # read_csv params
    params = ["csv_path"]
    params.append(f"sep={repr(settings.sep)}" if settings.sep is not None else "sep=','")
    if settings.quotechar is not None:
        params.append(f"quotechar={repr(settings.quotechar)}")
    if settings.escapechar is not None and settings.escapechar != settings.quotechar:
        params.append(f"escapechar={repr(settings.escapechar)}")
    params.append("header=0" if settings.header else "header=None")
    if settings.encoding:
        params.append(f"encoding={repr(settings.encoding)}")

    lines.append(f"df = pd.read_csv({', '.join(params)})")

    # Publish to context
    lines.append("# publish to context")
    lines.extend(context_assignment_lines(node_id, out_ports))

    return lines


def generate_ipynb_code(node_id: str, node_dir: Optional[str], out_ports: List[str]) -> str:
    """
    Return the code cell text for the notebook workbook (single string).
    """
    body = generate_py_body(node_id, node_dir, out_ports)
    return "\n".join(body) + "\n"
