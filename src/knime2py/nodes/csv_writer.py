#!/usr/bin/env python3

"""
CSV Writer module.

Overview
----------------------------
This module generates Python code to write a pandas DataFrame to a CSV file based on 
settings parsed from settings.xml. It fits into the knime2py generator pipeline as a 
node that handles CSV output.

Runtime Behavior
----------------------------
Inputs:
- Reads a single DataFrame from the context using the key format 'src_id:in_port'.

Outputs:
- Writes the DataFrame to a specified CSV file path, which is determined by settings.xml 
  and mapped to the output context.

Key algorithms or mappings:
- The module maps KNIME writer options to pandas.to_csv parameters, including handling 
  of separators, quote characters, and header options.

Edge Cases
----------------------------
The code implements safeguards for missing output paths, defaulting to a placeholder if 
not found. It also handles NaN values based on the na_rep setting.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries: pandas, lxml. These 
dependencies are required for the generated code, not for this module itself.

Usage
----------------------------
This module is typically invoked by the knime2py emitter when processing a CSV Writer 
node. An example of expected context access is:
```python
df = context['source_id:1']
```

Node Identity
----------------------------
KNIME factory id: org.knime.base.node.io.filehandling.csv.writer.CSVWriter2NodeFactory.

Configuration
----------------------------
The settings are encapsulated in the `CSVWriterSettings` dataclass, which includes 
important fields such as:
- path: The output file path (default: None).
- sep: The separator used in the CSV file (default: ",").
- quotechar: The character used for quoting (default: '"').
- header: Whether to write the header (default: True).
- encoding: The file encoding (default: "utf-8").
- na_rep: Representation for NaN values (default: None).
- include_index: Whether to include the DataFrame index (default: False).

The parse_csv_writer_settings function extracts these values from settings.xml, 
handling both LOCAL and RELATIVE paths.

Limitations
----------------------------
The module does not implement options for appending to existing files or advanced 
error handling beyond basic path resolution.

References
----------------------------
For more information, refer to the KNIME documentation and the hub at 
https://hub.knime.com/knime/extensions/org.knime.features.base/latest/org.knime.base.node.io.filehandling.csv.writer.CSVWriter2NodeFactory.
"""

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
    quote_mode: str = "MINIMAL"
    keep_trailing_zero_in_decimals: bool = True


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
    quote_mode = first(root, ".//*[local-name()='entry' and @key='quote_mode']/@value") or "MINIMAL"
    keep_trailing_zero = bool_from_value(
        first(root, ".//*[local-name()='entry' and @key='keep_trailing_zero_in_decimals']/@value")
    )
    if keep_trailing_zero is None:
        keep_trailing_zero = True

    return CSVWriterSettings(
        path=out_path,
        sep=sep,
        quotechar=quotechar,
        header=header,
        encoding=enc,
        na_rep=na_rep,
        include_index=include_index,
        quote_mode=quote_mode,
        keep_trailing_zero_in_decimals=keep_trailing_zero,
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
    return ["from pathlib import Path", "import csv", "import pandas as pd"]

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


def generate_py_body(node_id: str, node_dir: Optional[str], in_ports: List[tuple[str, str]]) -> List[str]:
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

    lines.extend(
        [
            "df = df.copy()",
            "for _col in df.select_dtypes(include=['datetime', 'datetimetz']).columns:",
            "    df[_col] = df[_col].dt.strftime('%Y-%m-%dT%H:%M')",
            "def _k2p_format_timedelta(_value):",
            "    if pd.isna(_value):",
            "        return pd.NA",
            "    _total_minutes = int(pd.Timedelta(_value).total_seconds() // 60)",
            "    if _total_minutes == 0:",
            "        return 'PT0S'",
            "    _sign = -1 if _total_minutes < 0 else 1",
            "    _hours, _minutes = divmod(abs(_total_minutes), 60)",
            "    if _sign < 0:",
            "        if _hours and _minutes:",
            "            return f'PT-{_hours}H-{_minutes}M'",
            "        if _hours:",
            "            return f'PT-{_hours}H'",
            "        return f'PT-{_minutes}M'",
            "    if _hours and _minutes:",
            "        return f'PT{_hours}H{_minutes}M'",
            "    if _hours:",
            "        return f'PT{_hours}H'",
            "    return f'PT{_minutes}M'",
            "for _col in df.select_dtypes(include=['timedelta']).columns:",
            "    df[_col] = df[_col].map(_k2p_format_timedelta)",
        ]
    )

    if not settings.keep_trailing_zero_in_decimals:
        lines.extend(
            [
                "for _col in df.select_dtypes(include=['float']).columns:",
                "    _series = df[_col].dropna()",
                "    if not _series.empty and ((_series % 1) == 0).all():",
                "        df[_col] = df[_col].astype('Int64')",
            ]
        )

    # Build to_csv kwargs
    quote_mode = (settings.quote_mode or "MINIMAL").strip().upper()
    use_missing_sentinel = quote_mode == "STRINGS_ONLY" and (settings.na_rep is None or settings.na_rep == "")
    if use_missing_sentinel:
        lines.extend(
            [
                "_k2p_missing_sentinel = '__K2P_MISSING_FIELD_9b1f6f7c__'",
                "df = df.astype('object').mask(df.isna(), _k2p_missing_sentinel)",
            ]
        )

    quoting = {
        "STRINGS_ONLY": "csv.QUOTE_NONNUMERIC",
        "ALL": "csv.QUOTE_ALL",
        "NONE": "csv.QUOTE_NONE",
        "MINIMAL": "csv.QUOTE_MINIMAL",
    }.get(quote_mode)
    kw_parts = [
        _fmt_kw("sep", settings.sep or ","),
        _fmt_kw("quotechar", settings.quotechar or '"'),
        _fmt_kw("header", bool(settings.header)),
        _fmt_kw("encoding", settings.encoding or "utf-8"),
        _fmt_kw("na_rep", settings.na_rep),          # may be '' (empty string)
        _fmt_kw("index", bool(settings.include_index)),
    ]
    if quoting:
        kw_parts.append(f"quoting={quoting}")
    kw_str = ", ".join(p for p in kw_parts if p is not None)

    lines.append(f"df.to_csv(out_path, {kw_str})")
    if use_missing_sentinel:
        encoding = settings.encoding or "utf-8"
        lines.extend(
            [
                f"_k2p_csv_text = out_path.read_text(encoding={encoding!r})",
                "_k2p_csv_text = _k2p_csv_text.replace('\"' + _k2p_missing_sentinel + '\"', '')",
                f"out_path.write_text(_k2p_csv_text, encoding={encoding!r})",
            ]
        )
    return lines



def get_name() -> str:
    """Return name of the node in KNIME workflow."""
    return "CSV Writer"


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
