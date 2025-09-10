#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from lxml import etree as ET
from ..xml_utils import XML_PARSER  # project helper (ok)
from .node_utils import *

CSV_FACTORY = "org.knime.base.node.io.filehandling.csv.reader.CSVTableReaderNodeFactory"

def can_handle(node_type: Optional[str]) -> bool:
    """Return True if this generator supports the node factory."""
    if not node_type:
        return False
    return node_type.endswith(".CSVTableReaderNodeFactory")

def _build_pandas_dtype_map(root: ET._Element) -> dict:
    spec = extract_table_spec_types(root)
    d = {}
    for col, jcls in (spec or {}).items():
        pdtype = java_to_pandas_dtype(jcls)
        if pdtype:
            d[col] = pdtype
    return d

@dataclass
class CSVReaderSettings:
    path: Optional[str] = None
    sep: Optional[str] = None
    quotechar: Optional[str] = None
    escapechar: Optional[str] = None
    header: Optional[bool] = None
    encoding: Optional[str] = None
    pandas_dtypes: dict = field(default_factory=dict)


# ----------------------------
# Settings.xml → CSVReaderSettings
# ----------------------------

from pathlib import Path
from typing import Optional
from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import first  # you already have this

def _resolve_reader_path(root: ET._Element, node_dir: Path) -> Optional[str]:
    """
    Resolve the CSV path from settings.xml. Supports:
      - LOCAL: absolute path is used as-is
      - RELATIVE + knime.workflow: path is relative to the workflow directory
    """
    path_cfg = first_el(root, ".//*[local-name()='config' and @key='path']")
    if path_cfg is None:
        return None

    raw_path = first(path_cfg, ".//*[local-name()='entry' and @key='path']/@value")
    fs_type  = first(path_cfg, ".//*[local-name()='entry' and @key='file_system_type']/@value")
    spec     = first(path_cfg, ".//*[local-name()='entry' and @key='file_system_specifier']/@value")

    if not raw_path:
        return None

    # If parse_csv_reader_settings was given the NODE folder (has settings.xml),
    # then the workflow folder is its parent; otherwise assume node_dir itself is the workflow folder.
    node_has_settings = (node_dir / "settings.xml").exists()
    workflow_dir = node_dir.parent if node_has_settings else node_dir

    try:
        p = Path(raw_path)
        if (fs_type or "").upper() == "LOCAL" or p.is_absolute():
            return str(p.expanduser().resolve())

        if (fs_type or "").upper() == "RELATIVE" and (spec or "").lower() == "knime.workflow":
            return str((workflow_dir / raw_path).expanduser().resolve())

        # Fallback: treat as relative to workflow_dir
        return str((workflow_dir / raw_path).expanduser().resolve())
    except Exception:
        # Last-ditch: just return the raw string
        return raw_path


def parse_csv_reader_settings(node_dir: Path) -> CSVReaderSettings:
    """
    Read <node_dir>/settings.xml and extract csv path & common options.
    Handles absolute LOCAL paths and RELATIVE paths anchored at the workflow directory.
    """
    settings = node_dir / "settings.xml"
    if not settings.exists():
        return CSVReaderSettings()

    root = ET.parse(str(settings), parser=XML_PARSER).getroot()

    # New: resolve path properly (LOCAL vs RELATIVE knime.workflow)
    file_path = _resolve_reader_path(root, node_dir)

    # Keep existing extractors for the rest
    sep = extract_csv_sep(root) or ","
    quotechar = extract_csv_quotechar(root) or '"'
    escapechar = extract_csv_escapechar(root)
    header = extract_csv_header_reader(root)
    if header is None:
        header = True
    enc = extract_csv_encoding(root) or "utf-8"
    pandas_dtypes = _build_pandas_dtype_map(root)

    return CSVReaderSettings(
        path=file_path,
        sep=sep,
        quotechar=quotechar,
        escapechar=escapechar,
        header=header,
        encoding=enc,
        pandas_dtypes=pandas_dtypes,
    )



# ----------------------------
# Code generators
# ----------------------------

def generate_imports():
    return ["from pathlib import Path", "import pandas as pd"]


def generate_py_body(node_id: str, node_dir: Optional[str], out_ports: List[str]) -> List[str]:
    """
    Emit body lines for a CSV Reader node that reads a CSV into `df`
    and publishes it to the provided context out_ports.
    """
    ndir = Path(node_dir) if node_dir else None
    settings = parse_csv_reader_settings(ndir) if ndir else CSVReaderSettings()

    lines: List[str] = []
    lines.append("# https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
                 "org.knime.base.node.io.filehandling.csv.reader.CSVTableReaderNodeFactory")

    # Path
    if settings.path:
        lines.append(f"csv_path = Path(r\"{settings.path}\")")
    else:
        lines.append("# WARNING: CSV path not found in settings.xml. Please set manually:")
        lines.append("csv_path = Path('path/to/input.csv')")

    # ---- Build read_csv kwargs ----
    # sep / quote / encoding
    sep_arg    = repr(settings.sep if settings.sep is not None else ",")
    quote_arg  = repr(settings.quotechar if settings.quotechar is not None else '"')
    enc_arg    = repr(settings.encoding if settings.encoding else "utf-8")

    # header: pandas expects 0 (row index) when file has a header, else None
    header_has = True if settings.header is None else bool(settings.header)
    header_arg = "0" if header_has else "None"

    # dtype mapping (optional)
    dtype_arg = None
    if getattr(settings, "pandas_dtypes", None):
        # literal dict is fine here; we rely on small sets of columns
        dtype_arg = repr(settings.pandas_dtypes)

    # Assemble call
    kwargs = [f"sep={sep_arg}", f"quotechar={quote_arg}", f"header={header_arg}", f"encoding={enc_arg}"]
    if dtype_arg:
        kwargs.append(f"dtype={dtype_arg}")

    lines.append(f"df = pd.read_csv(csv_path, {', '.join(kwargs)})")

    # Publish to context
    for line in context_assignment_lines(node_id, out_ports):
        lines.append(line)

    return lines


def generate_ipynb_code(node_id: str, node_dir: Optional[str], out_ports: List[str]) -> str:
    """
    Return the code cell text for the notebook workbook (single string).
    """
    body = generate_py_body(node_id, node_dir, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    if not (ntype and can_handle(ntype)):
        return None
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in outgoing]
    node_lines = generate_py_body(nid, npath, out_ports)

    found, body = split_out_imports(node_lines)
    explicit = collect_module_imports(generate_imports) 
    imports = sorted(set(found).union(explicit))
    return imports, body