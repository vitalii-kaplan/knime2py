#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, normalize_in_ports, split_out_imports


FACTORY = "org.knime.database.node.io.reader.query.DBQueryReaderNodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.database/latest/"
    "org.knime.database.node.io.reader.query.DBQueryReaderNodeFactory"
)


@dataclass
class DBQueryReaderSettings:
    sql_statement: Optional[str] = None


def parse_db_query_reader_settings(node_dir: Optional[Path]) -> DBQueryReaderSettings:
    if not node_dir:
        return DBQueryReaderSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return DBQueryReaderSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    return DBQueryReaderSettings(
        sql_statement=first(model, "./*[local-name()='entry' and @key='sql_statement']/@value") if model is not None else None
    )


def generate_imports() -> list[str]:
    return ["import pandas as pd", "import sqlite3"]


def _emit_query_code(settings: DBQueryReaderSettings) -> list[str]:
    return [
        f"_sql = {repr(settings.sql_statement or '')}",
        "if not _sql.strip():",
        "    out_df = pd.DataFrame()",
        "elif db_connection.get('dialect') == 'sqlite':",
        "    _sqlite_path = db_connection.get('sqlite_path')",
        "    if not _sqlite_path:",
        "        raise ValueError('SQLite DB Query Reader requires sqlite_path in db connector descriptor')",
        "    with sqlite3.connect(_sqlite_path) as _conn:",
        "        out_df = pd.read_sql_query(_sql, _conn)",
        "else:",
        "    raise NotImplementedError(f\"DB Query Reader only supports sqlite descriptors currently, got {db_connection.get('dialect')!r}\")",
    ]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: list[tuple[str, str]],
    out_ports: Optional[list[str]] = None,
) -> list[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_db_query_reader_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines = [
        f"# {HUB_URL}",
        f"db_connection = context['{src_id}:{src_port}']",
    ]
    lines.extend(_emit_query_code(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "DB Query Reader"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
