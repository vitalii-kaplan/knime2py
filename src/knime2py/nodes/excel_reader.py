#!/usr/bin/env python3

"""
Excel Reader module.

Overview
----------------------------
This module reads an Excel sheet into a pandas DataFrame using options parsed from
settings.xml. It fits into the knime2py generator pipeline by providing a way to
convert KNIME Excel reader node configurations into Python code.

Runtime Behavior
----------------------------
Inputs:
- Reads a single DataFrame from an Excel file specified in the settings.

Outputs:
- Writes the resulting DataFrame to the context, mapped to the specified output ports.

Key algorithms or mappings:
- Maps KNIME options such as sheet selection, header presence, column and row ranges,
  and data types to their pandas equivalents.

Edge Cases
----------------------------
The code implements safeguards for empty or constant columns, handles NaNs, and provides
fallback paths for missing settings.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries: pandas, lxml. These
dependencies are required by the generated code, not by this module.

Usage
----------------------------
This module is typically invoked by the knime2py emitter for Excel reader nodes.
An example of expected context access is:
```
context['output_port_1'] = df
```

Node Identity
----------------------------
KNIME factory id: FACTORY
No special flags are defined for this node.

Configuration
----------------------------
The settings are encapsulated in the `ExcelReaderSettings` dataclass, which includes
the following important fields:
- path: The file path to the Excel file (default: None).
- sheet: The sheet to read (default: 0).
- header: The row number to use as the column names (default: 0).
- usecols: The columns to read (default: None).
- skiprows: The number of rows to skip at the start (default: None).
- nrows: The number of rows to read (default: None).
- replace_empty_with_na: Whether to replace empty strings with NaN (default: False).
- pandas_dtypes: A mapping of column names to pandas data types (default: empty dict).

The `parse_excel_reader_settings` function extracts these values from settings.xml
using XPath queries and provides fallbacks for missing entries.

Limitations
----------------------------
The module does not support hidden rows/columns, formula re-evaluation, and 15-digit
precision, which are not directly supported by pandas.

References
----------------------------
For more information, refer to the KNIME documentation and the following URL:
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/
org.knime.ext.poi3.node.io.filehandling.excel.reader.ExcelTableReaderNodeFactory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (
    first,
    first_el,
    iter_entries,
    extract_table_spec_types,
    java_to_pandas_dtype,
    resolve_reader_path,
    collect_module_imports,
    split_out_imports,
    context_assignment_lines,
)

FACTORY = "org.knime.ext.poi3.node.io.filehandling.excel.reader.ExcelTableReaderNodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.ext.poi3.node.io.filehandling.excel.reader.ExcelTableReaderNodeFactory"
)


# -----------------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------------

def _col_letter(s: Optional[str]) -> Optional[str]:
    """Normalize an Excel column letter like 'A', 'AB', return None if empty/invalid."""
    if not s:
        return None
    v = "".join(ch for ch in str(s).strip() if ch.isalpha())
    return v.upper() or None

def _to_int(s: Optional[str]) -> Optional[int]:
    """Convert a string to an integer, returning None if conversion fails."""
    try:
        return int(str(s).strip())
    except Exception:
        return None

def _build_pandas_dtype_map(root: ET._Element) -> Dict[str, str]:
    """Map KNIME table spec java class → pandas dtype (nullable where possible)."""
    spec = extract_table_spec_types(root)
    out: Dict[str, str] = {}
    for col, jcls in (spec or {}).items():
        pdtype = java_to_pandas_dtype(jcls)
        if pdtype:
            out[col] = pdtype
    return out


# -----------------------------------------------------------------------------------
# settings.xml → ExcelReaderSettings
# -----------------------------------------------------------------------------------

@dataclass
class ExcelReaderSettings:
    path: Optional[str] = None
    sheet: Any = 0  # 0 | int | str
    header: Optional[int] = 0
    usecols: Optional[str] = None
    skiprows: Optional[int] = None
    nrows: Optional[int] = None
    replace_empty_with_na: bool = False
    pandas_dtypes: Dict[str, str] = field(default_factory=dict)


def parse_excel_reader_settings(node_dir: Optional[Path]) -> ExcelReaderSettings:
    """Parse settings from settings.xml and return an ExcelReaderSettings object."""
    if not node_dir:
        return ExcelReaderSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return ExcelReaderSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    # File path resolution (LOCAL/RELATIVE knime.workflow)
    path = resolve_reader_path(root, node_dir)

    # Sheet selection
    sel = (first(root, ".//*[local-name()='entry' and @key='sheet_selection']/@value") or "").upper()
    sheet_name = first(root, ".//*[local-name()='entry' and @key='sheet_name']/@value")
    sheet_idx = _to_int(first(root, ".//*[local-name()='entry' and @key='sheet_index']/@value"))

    if sel == "NAME" and (sheet_name or "").strip():
        sheet = sheet_name
    elif sel == "INDEX" and sheet_idx is not None:
        sheet = int(sheet_idx)
    else:
        sheet = 0  # FIRST

    # Header
    has_header = first(root, ".//*[local-name()='entry' and @key='table_contains_column_names']/@value")
    has_header = (str(has_header or "").strip().lower() in {"true", "1", "yes", "y"})
    header_row_1 = _to_int(first(root, ".//*[local-name()='entry' and @key='column_names_row_number']/@value")) or 1
    header = (header_row_1 - 1) if has_header else None

    # Column range → usecols (A:D)
    c_from = _col_letter(first(root, ".//*[local-name()='entry' and @key='read_from_column']/@value"))
    c_to   = _col_letter(first(root, ".//*[local-name()='entry' and @key='read_to_column']/@value"))
    usecols = None
    if c_from and c_to:
        usecols = f"{c_from}:{c_to}"

    # Row range → skiprows / nrows (best-effort)
    r_from = _to_int(first(root, ".//*[local-name()='entry' and @key='read_from_row']/@value"))
    r_to   = _to_int(first(root, ".//*[local-name()='entry' and @key='read_to_row']/@value"))
    skiprows = None
    nrows = None
    if r_from and r_from > 1:
        # Pandas applies header AFTER skiprows; that's fine—header is relative to post-skip.
        skiprows = r_from - 1
    if r_from and r_to and r_to >= r_from:
        nrows = (r_to - r_from + 1)

    # Advanced: replace empty strings with missings?
    replace_empty = first(root, ".//*[local-name()='entry' and @key='replace_empty_strings_with_missings']/@value")
    replace_empty = (str(replace_empty or "").strip().lower() in {"true", "1", "yes", "y"})

    # Dtypes from spec
    pandas_dtypes = _build_pandas_dtype_map(root)

    return ExcelReaderSettings(
        path=path,
        sheet=sheet,
        header=header,
        usecols=usecols,
        skiprows=skiprows,
        nrows=nrows,
        replace_empty_with_na=replace_empty,
        pandas_dtypes=pandas_dtypes,
    )


# -----------------------------------------------------------------------------------
# Code generators
# -----------------------------------------------------------------------------------

def generate_imports():
    """Generate a list of import statements required for the Excel reader."""
    return ["from pathlib import Path", "import pandas as pd"]


def generate_py_body(node_id: str, node_dir: Optional[str], out_ports: List[str]) -> List[str]:
    """Generate the Python code body for reading an Excel file based on the provided settings."""
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_excel_reader_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Path
    if cfg.path:
        lines.append(f"xls_path = Path(r\"{cfg.path}\")")
    else:
        lines.append("# WARNING: Excel path not found in settings.xml. Please set manually:")
        lines.append("xls_path = Path('path/to/input.xlsx')")

    # Build read_excel kwargs
    sheet_repr = repr(cfg.sheet) if isinstance(cfg.sheet, str) else (str(int(cfg.sheet)) if isinstance(cfg.sheet, int) else "0")
    header_repr = "None" if cfg.header is None else str(int(cfg.header))

    kwargs = [f"sheet_name={sheet_repr}", f"header={header_repr}"]

    if cfg.usecols:
        kwargs.append(f"usecols={repr(cfg.usecols)}")
    if cfg.skiprows is not None:
        kwargs.append(f"skiprows={int(cfg.skiprows)}")
    if cfg.nrows is not None:
        kwargs.append(f"nrows={int(cfg.nrows)}")
    if cfg.pandas_dtypes:
        kwargs.append(f"dtype={repr(cfg.pandas_dtypes)}")

    # Read
    lines.append(f"df = pd.read_excel(xls_path, {', '.join(kwargs)})")

    # Replace empty strings with NA if requested
    if cfg.replace_empty_with_na:
        lines.append("df = df.replace({\"\": pd.NA})")

    # Publish to context
    for line in context_assignment_lines(node_id, out_ports):
        lines.append(line)

    # Notes for not-yet-mapped features
    lines.append("# NOTE: Hidden rows/columns, formula re-evaluation, and 15-digit precision are not directly supported.")
    return lines


def generate_ipynb_code(node_id: str, node_dir: Optional[str], out_ports: List[str]) -> str:
    """Generate the complete IPython notebook code for the Excel reader node."""
    body = generate_py_body(node_id, node_dir, out_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Returns (imports, body_lines) for Excel Reader.
    This is a source node: no incoming ports are required. We emit to all outgoing ports.
    """
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]

    node_lines = generate_py_body(nid, npath, out_ports)
    found_imports, body = split_out_imports(node_lines)
    explicit_imports = collect_module_imports(generate_imports)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
