#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, split_out_imports


FACTORY = "org.knime.database.extension.postgres.node.connector.PostgreSQLDBConnectorNodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.database/latest/"
    "org.knime.database.extension.postgres.node.connector.PostgreSQLDBConnectorNodeFactory"
)


@dataclass
class PostgreSQLConnectorSettings:
    db_type: Optional[str] = None
    db_dialect: Optional[str] = None
    db_driver: Optional[str] = None
    username: Optional[str] = None
    auth_type: Optional[str] = None
    host: Optional[str] = None
    port: Optional[str] = None
    database_name: Optional[str] = None


def parse_postgresql_connector_settings(node_dir: Optional[Path]) -> PostgreSQLConnectorSettings:
    if not node_dir:
        return PostgreSQLConnectorSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return PostgreSQLConnectorSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    session = first_el(model, "./*[local-name()='config' and @key='session_info']") if model is not None else None
    auth = first_el(model, "./*[local-name()='config' and @key='authentication']") if model is not None else None
    postgres = first_el(model, "./*[local-name()='config' and @key='postgres-connection']") if model is not None else None

    return PostgreSQLConnectorSettings(
        db_type=first(session, "./*[local-name()='entry' and @key='db_type']/@value") if session is not None else None,
        db_dialect=first(session, "./*[local-name()='entry' and @key='db_dialect']/@value") if session is not None else None,
        db_driver=first(session, "./*[local-name()='entry' and @key='db_driver']/@value") if session is not None else None,
        username=first(auth, "./*[local-name()='entry' and @key='username']/@value") if auth is not None else None,
        auth_type=first(auth, "./*[local-name()='entry' and @key='selectedType']/@value") if auth is not None else None,
        host=first(postgres, "./*[local-name()='entry' and @key='host']/@value") if postgres is not None else None,
        port=first(postgres, "./*[local-name()='entry' and @key='port']/@value") if postgres is not None else None,
        database_name=(
            first(postgres, "./*[local-name()='entry' and @key='database_name']/@value") if postgres is not None else None
        ),
    )


def generate_imports() -> list[str]:
    return []


def _sqlalchemy_url(settings: PostgreSQLConnectorSettings) -> Optional[str]:
    if not settings.host or not settings.database_name:
        return None
    auth = f"{settings.username}@" if settings.username else ""
    port = f":{settings.port}" if settings.port else ""
    return f"postgresql+psycopg://{auth}{settings.host}{port}/{settings.database_name}"


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: list[tuple[str, str]],
    out_ports: Optional[list[str]] = None,
) -> list[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_postgresql_connector_settings(ndir)
    dialect = (settings.db_dialect or settings.db_type or "postgres").lower()

    descriptor = {
        "kind": "database",
        "dialect": dialect,
        "db_type": settings.db_type,
        "db_driver": settings.db_driver,
        "jdbc_url": None,
        "username": settings.username,
        "auth_type": settings.auth_type,
        "host": settings.host,
        "port": settings.port,
        "database_name": settings.database_name,
        "sqlalchemy_url": _sqlalchemy_url(settings),
        "source": "settings.xml",
    }

    lines = [
        f"# {HUB_URL}",
        f"db_connection = {repr(descriptor)}",
    ]

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = db_connection")
    return lines


def get_name() -> str:
    return "PostgreSQL Connector"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, [], out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
