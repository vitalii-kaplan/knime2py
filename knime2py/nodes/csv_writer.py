#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET
from ..xml_utils import XML_PARSER  # project helper is OK here


CSV_FACTORY = "org.knime.base.node.io.filehandling.csv.writer.CSVWriter2NodeFactory"


def can_handle(node_type: Optional[str]) -> bool:
    """Return True if this generator supports the node factory."""
    if not node_type:
        return False
    # KNIME 5.x CSV writer
    return node_type.endswith(".CSVWriter2NodeFactory")


@dataclass
class CSVWriterSettings:
    out_path: Optional[str] = None
    sep: Optional[str] = None
    quotechar: Optional[str] = None
    escapechar: Optional[str] = None
    header: Optional[bool] = None
    encoding: Optional[str] = None
    line_terminator: Optional[str] = None
    append: Optional[bool] = None
    overwrite: Optional[bool] = None
    na_rep: Optional[str] = None


# ----------------------------
# XML helpers
# ----------------------------

def _first(root: ET._Element, xpath: str) -> Optional[str]:
    vals = root.xpath(xpath)
    if vals:
        return (vals[0] or "").strip()
    return None


def _all(root: ET._Element, xpath: str) -> List[str]:
    return [(v or "").strip() for v in root.xpath(xpath)]


def _normalize_delim(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    v = raw.strip()
    if len(v) == 1:
        return v
    up = v.upper()
    if up in {"TAB", "\\T", "CTRL-I"}:
        return "\t"
    if up in {"COMMA"}:
        return ","
    if up in {"SEMICOLON", "SEMI", "SC"}:
        return ";"
    if up in {"SPACE"}:
        return " "
    if up in {"PIPE"}:
        return "|"
    if v == "\\t":
        return "\t"
    return v or None


def _normalize_char(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    v = raw.strip()
    if v.upper() in {"", "NONE", "NULL"}:
        return None
    if v == "&quot;":
        return '"'
    if v == "&apos;":
        return "'"
    return v[:1] if len(v) >= 1 else None


def _bool_from_value(v: Optional[str]) -> Optional[bool]:
    if v is None:
        return None
    up = v.strip().lower()
    if up in {"true", "1", "yes", "y"}:
        return True
    if up in {"false", "0", "no", "n"}:
        return False
    return None


def _looks_like_path(s: str) -> bool:
    if not s:
        return False
    if s.lower().startswith(("file:", "s3:", "hdfs:", "abfss:", "http://", "https://")):
        return True
    if s.endswith(".csv"):
        return True
    if "/" in s or "\\" in s:
        return True
    return False


# ----------------------------
# settings.xml -> CSVWriterSettings
# ----------------------------

def parse_csv_writer_settings(node_dir: Path) -> CSVWriterSettings:
    """
    Extract CSV writer options from <node_dir>/settings.xml.
    Heuristic, robust across KNIME 5.x variants.
    """
    settings = node_dir / "settings.xml"
    if not settings.exists():
        return CSVWriterSettings()

    root = ET.parse(str(settings), parser=XML_PARSER).getroot()

    # Output path: look for common keys
    out_candidates = _all(
        root,
        "(.//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'path')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'file')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'location')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'dest')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'output')]/@value)"
    )
    out_path = next((p for p in out_candidates if _looks_like_path(p)), None)

    # Separator
    delim_raw = _first(
        root,
        "(.//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'delim')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','lowercase'),'separator')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'separator')]/@value)"
    )
    sep = _normalize_delim(delim_raw) or ","

    # Quote / escape
    quote_raw = _first(
        root,
        ".//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'quote')]/@value"
    )
    quotechar = _normalize_char(quote_raw) or '"'

    esc_raw = _first(
        root,
        ".//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'escape')]/@value"
    )
    escapechar = _normalize_char(esc_raw)

    # Header flag (write column header)
    header_raw = _first(
        root,
        "(.//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'header')]/@value"
        " | .//*[local-name()='entry' and @key='header']/@value)"
    )
    header = _bool_from_value(header_raw)
    if header is None:
        header = True

    # Encoding
    enc = _first(
        root,
        "(.//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'charset')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'encoding')]/@value)"
    ) or "utf-8"

    # Line terminator / newline
    lt = _first(
        root,
        "(.//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'line')"
        " and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'separator')]/@value"
        " | .//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'lineterminator')]/@value)"
    )

    # Append / overwrite hints
    append = _bool_from_value(_first(root, ".//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'append')]/@value"))
    overwrite = _bool_from_value(_first(root, ".//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'overwrite')]/@value"))

    # Missing value representation
    na_rep = _first(root, ".//*[local-name()='entry' and contains(translate(@key,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'missing')]/@value")

    return CSVWriterSettings(
        out_path=out_path,
        sep=sep,
        quotechar=quotechar,
        escapechar=escapechar,
        header=header,
        encoding=enc,
        line_terminator=lt,
        append=append,
        overwrite=overwrite,
        na_rep=na_rep,
    )


# ----------------------------
# Code generators
# ----------------------------

def _context_read_lines(in_ports: List[str]) -> List[str]:
    lines: List[str] = []
    if in_ports:
        # prefer the first declared input
        key = in_ports[0]
        lines.append(f"df = context[{key!r}]")
    else:
        lines.append("# WARNING: no inputs detected; please set `df` manually")
        lines.append("df = None  # replace with your DataFrame")
    return lines


def generate_py_body(node_id: str, node_dir: Optional[str], in_ports: List[str]) -> List[str]:
    """
    Return the *body* lines to place inside the function for this node in the .py workbook.
    """
    ndir = Path(node_dir) if node_dir else None
    settings = parse_csv_writer_settings(ndir) if ndir else CSVWriterSettings()

    lines: List[str] = []
    lines.append("# https://hub.knime.com/knime/extensions/org.knime.features.base/latest/" + CSV_FACTORY)
    lines.append("from pathlib import Path")
    lines.append("import pandas as pd")

    # Read DF from context
    lines += _context_read_lines(in_ports)

    # Output path
    if settings.out_path:
        lines.append(f"out_path = Path(r\"{settings.out_path}\")")
    else:
        lines.append("# WARNING: CSV output path not found in settings.xml. Please set manually:")
        lines.append("out_path = Path('output.csv')")

    # Build to_csv params safely (avoid backslashes in f-string expressions)
    params: List[str] = ["out_path"]
    # sep
    sep = settings.sep if settings.sep is not None else ","
    params.append(f"sep={sep!r}")
    # quotechar (default '"' if None)
    q = settings.quotechar if settings.quotechar is not None else '"'
    params.append(f"quotechar={q!r}")
    # escapechar only if present and different from quotechar
    if settings.escapechar is not None and settings.escapechar != q:
        params.append(f"escapechar={settings.escapechar!r}")
    # header flag
    params.append("header=True" if (settings.header is None or settings.header) else "header=False")
    # encoding
    if settings.encoding:
        params.append(f"encoding={settings.encoding!r}")
    # line terminator
    if settings.line_terminator:
        params.append(f"lineterminator={settings.line_terminator!r}")
    # na_rep
    if settings.na_rep:
        params.append(f"na_rep={settings.na_rep!r}")
    # Never write the pandas index by default
    params.append("index=False")
    # mode (append/overwrite hints)
    if settings.append:
        params.append("mode='a'")
    elif settings.overwrite:
        params.append("mode='w'")

    lines.append(f"df.to_csv({', '.join(params)})")
    return lines


def generate_ipynb_code(node_id: str, node_dir: Optional[str], in_ports: List[str]) -> str:
    """
    Return the code cell text for the notebook workbook (single string).
    """
    body = generate_py_body(node_id, node_dir, in_ports)
    return "\n".join(body) + "\n"
