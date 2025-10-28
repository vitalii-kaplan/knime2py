#!/usr/bin/env python3

####################################################################################################
#
# Excel Writer
#
# Writes a pandas DataFrame to an Excel file using options parsed from settings.xml.
# Resolves LOCAL/RELATIVE (knime.workflow) paths and maps common KNIME writer options to
# pandas.ExcelWriter / DataFrame.to_excel.
# Covered mappings (KNIME → pandas):
#   • Path: LOCAL & RELATIVE (knime.workflow) via resolve_reader_path()
#   • Format: excel_format (XLSX) → engine='openpyxl' (others not implemented)
#   • Sheet names: sheet_names[*] → one sheet per entry (writes same df to each)
#   • If sheet exists: {FAIL, REPLACE, NEW} → if_sheet_exists={'error','replace','new'}
#   • If path exists: {fail, overwrite, append} → pre-check + writer mode {'w','a'}
#   • Row key: write_row_key → index=bool
#   • Column header: write_column_header → header=bool
#   • Prob. header on append: skip_column_header_on_append (best-effort; see Note)
#   • Replace missings: replace_missings + missing_value_pattern → na_rep=<string>
#
#   - Only XLSX is supported (engine='openpyxl'). Legacy XLS (xls) is not implemented.
#   - KNIME-style row-wise append into an existing sheet is not fully replicated. Pandas
#     does not support true “append to bottom” without custom openpyxl manipulation.
#   - We honor if_sheet_exists and header flags but do not append rows.
#   - Auto-size columns, print layout, formula evaluation, and “open file after exec”
#     are not supported.
#
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import (
    first,
    first_el,
    iter_entries,
    normalize_in_ports,
    resolve_reader_path,
    collect_module_imports,
    split_out_imports,
)

FACTORY = "org.knime.ext.poi3.node.io.filehandling.excel.writer.ExcelTableWriterNodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
    "org.knime.ext.poi3.node.io.filehandling.excel.writer.ExcelTableWriterNodeFactory"
)

# ---------------------------------------------------------------------
# Settings.xml → ExcelWriterSettings
# ---------------------------------------------------------------------

@dataclass
class ExcelWriterSettings:
    path: Optional[str] = None
    excel_format: str = "XLSX"
    sheet_names: List[str] = field(default_factory=lambda: ["Sheet1"])
    if_sheet_exists: str = "FAIL"      # FAIL | REPLACE | NEW
    if_path_exists: str = "overwrite"  # fail | overwrite | append
    write_row_key: bool = False
    write_column_header: bool = True
    skip_header_on_append: bool = False
    replace_missings: bool = False
    missing_value_pattern: Optional[str] = None
    create_missing_folders: bool = False


def _bool(v: Optional[str], default: bool = False) -> bool:
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
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_excel_writer_settings(node_dir: Optional[Path]) -> ExcelWriterSettings:
    """
    Parse the Excel writer settings from the settings.xml file.

    Args:
        node_dir (Optional[Path]): The directory containing the settings.xml file.

    Returns:
        ExcelWriterSettings: The parsed settings.
    """
    if not node_dir:
        return ExcelWriterSettings()

    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return ExcelWriterSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    # Resolve output path (LOCAL / RELATIVE knime.workflow)
    out_path = resolve_reader_path(root, node_dir)

    # Format (only XLSX supported here)
    excel_format = (first(root, ".//*[local-name()='entry' and @key='excel_format']/@value") or "XLSX").upper()

    # Sheet names
    sheet_names: List[str] = []
    for ent in root.xpath(".//*[local-name()='config' and @key='sheet_names']/*[local-name()='entry']"):
        k = ent.get("key") or ""
        if k.isdigit():
            val = (ent.get("value") or "").strip()
            if val:
                sheet_names.append(val)
    if not sheet_names:
        sheet_names = ["Sheet1"]

    # If sheet exists
    ise_raw = (first(root, ".//*[local-name()='entry' and @key='if_sheet_exists']/@value") or "").upper()
    if_sheet_exists = ise_raw if ise_raw in {"FAIL", "REPLACE", "NEW"} else "FAIL"

    # If path exists
    ipe_raw = (first(root, ".//*[local-name()='entry' and @key='if_path_exists']/@value") or "overwrite").lower()
    if_path_exists = ipe_raw if ipe_raw in {"fail", "overwrite", "append"} else "overwrite"

    # Headers / index
    write_row_key = _bool(first(root, ".//*[local-name()='entry' and @key='write_row_key']/@value"), False)
    write_column_header = _bool(first(root, ".//*[local-name()='entry' and @key='write_column_header']/@value"), True)
    skip_header_on_append = _bool(first(root, ".//*[local-name()='entry' and @key='skip_column_header_on_append']/@value"), False)

    # Missings
    replace_missings = _bool(first(root, ".//*[local-name()='entry' and @key='replace_missings']/@value"), False)
    missing_value_pattern = first(root, ".//*[local-name()='entry' and @key='missing_value_pattern']/@value")

    # Create folders?
    create_missing_folders = _bool(first(root, ".//*[local-name()='entry' and @key='create_missing_folders']/@value"), False)

    return ExcelWriterSettings(
        path=out_path,
        excel_format=excel_format,
        sheet_names=sheet_names,
        if_sheet_exists=if_sheet_exists,
        if_path_exists=if_path_exists,
        write_row_key=write_row_key,
        write_column_header=write_column_header,
        skip_header_on_append=skip_header_on_append,
        replace_missings=replace_missings,
        missing_value_pattern=missing_value_pattern if replace_missings else None,
        create_missing_folders=create_missing_folders,
    )


# ---------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------

def generate_imports():
    """
    Generate the necessary imports for the Excel writer.

    Returns:
        List[str]: A list of import statements.
    """
    # openpyxl is required for XLSX writing and append mode
    return ["from pathlib import Path", "import pandas as pd"]


def _map_if_sheet_exists(kn: str) -> str:
    """
    Map KNIME's sheet existence policy to pandas' equivalent.

    Args:
        kn (str): The KNIME sheet existence policy.

    Returns:
        str: The corresponding pandas policy.
    """
    m = {"FAIL": "error", "REPLACE": "replace", "NEW": "new"}
    return m.get(kn.upper(), "error")


def generate_py_body(node_id: str, node_dir: Optional[str], in_ports: List[object]) -> List[str]:
    """
    Generate the Python code body for the Excel writer.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The list of incoming ports.

    Returns:
        List[str]: The generated Python code lines.
    """
    ndir = Path(node_dir) if node_dir else None
    cfg = parse_excel_writer_settings(ndir)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    # Input df
    pairs = normalize_in_ports(in_ports)
    src_id, in_port = pairs[0]
    lines.append(f"df = context['{src_id}:{in_port}']")

    # Path
    if cfg.path:
        lines.append(f"out_path = Path(r\"{cfg.path}\")")
    else:
        lines.append("# WARNING: Excel output path not found in settings.xml. Set manually:")
        lines.append("out_path = Path('path/to/output.xlsx')")

    # Create folders if requested
    if cfg.create_missing_folders:
        lines.append("out_path.parent.mkdir(parents=True, exist_ok=True)")

    # If path exists policy
    lines.append("if out_path.exists():")
    if cfg.if_path_exists == "fail":
        lines.append("    raise FileExistsError(f'Excel Writer: path already exists: {out_path}')")
        writer_mode = "w"
    elif cfg.if_path_exists == "append":
        lines.append("    pass  # append mode")
        writer_mode = "a"
    else:
        # overwrite
        lines.append("    pass  # overwrite")
        writer_mode = "w"

    # Only XLSX supported
    if cfg.excel_format != "XLSX":
        lines.append(f"# NOTE: excel_format={cfg.excel_format!r} not supported; defaulting to XLSX.")

    # Pandas ExcelWriter options
    if_sheet_exists_pd = _map_if_sheet_exists(cfg.if_sheet_exists)
    lines.append(f"_if_sheet_exists = {repr(if_sheet_exists_pd)}")
    lines.append(f"_writer_mode = {repr(writer_mode)}  # 'w' overwrite, 'a' append")

    # Header/index flags
    # If appending and skipping header-on-append requested, best-effort: pass header=False.
    # True row-wise append is not implemented.
    if cfg.if_path_exists == "append" and cfg.skip_header_on_append:
        header_expr = "False"
    else:
        header_expr = "True" if cfg.write_column_header else "False"

    index_expr = "True" if cfg.write_row_key else "False"
    na_rep_expr = repr(cfg.missing_value_pattern) if cfg.missing_value_pattern is not None else "None"

    # Write the file
    lines.append("with pd.ExcelWriter(out_path, engine='openpyxl', mode=_writer_mode, if_sheet_exists=_if_sheet_exists) as _xw:")
    for sn in (cfg.sheet_names or ["Sheet1"]):
        sn_lit = repr(sn)
        lines.append(f"    df.to_excel(_xw, sheet_name={sn_lit}, header={header_expr}, index={index_expr}, na_rep={na_rep_expr})")

    # Unmapped features note
    lines.append("# NOTE: Auto-size columns, print layout, and formula evaluation are not supported in this export.")

    return lines


def generate_ipynb_code(node_id: str, node_dir: Optional[str], in_ports: List[object]) -> str:
    """
    Generate the code for a Jupyter notebook cell.

    Args:
        node_id (str): The ID of the node.
        node_dir (Optional[str]): The directory of the node.
        in_ports (List[object]): The list of incoming ports.

    Returns:
        str: The generated code for the notebook cell.
    """
    body = generate_py_body(node_id, node_dir, in_ports)
    return "\n".join(body) + "\n"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handle the Excel Writer node processing.

    Args:
        ntype: The type of the node.
        nid: The ID of the node.
        npath: The path of the node.
        incoming: The incoming connections.
        outgoing: The outgoing connections.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the imports and body lines for the node.
    """
    in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])] or [("UNKNOWN", "1")]
    node_lines = generate_py_body(nid, npath, in_ports)

    found_imports, body = split_out_imports(node_lines)
    explicit_imports = collect_module_imports(generate_imports)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
