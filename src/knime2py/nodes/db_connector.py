#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, first_el, split_out_imports


FACTORY = "org.knime.database.node.connector.generic.DBConnectorNodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.database/latest/"
    "org.knime.database.node.connector.generic.DBConnectorNodeFactory"
)


@dataclass
class DBConnectorSettings:
    db_type: Optional[str] = None
    db_dialect: Optional[str] = None
    db_driver: Optional[str] = None
    jdbc_url: Optional[str] = None
    username: Optional[str] = None
    auth_type: Optional[str] = None
    sqlite_path: Optional[str] = None
    source: str = "settings.xml"


def _repo_root_from_node_dir(node_dir: Path) -> Path:
    for candidate in [node_dir, *node_dir.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def _load_sidecar(node_dir: Path) -> dict[str, Any]:
    sidecar = node_dir / "knime2py.local.json"
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_sidecar_path(raw: str, node_dir: Path) -> str:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path.resolve())

    repo_candidate = _repo_root_from_node_dir(node_dir) / path
    if repo_candidate.exists():
        return str(repo_candidate.resolve())

    node_candidate = node_dir / path
    if node_candidate.exists():
        return str(node_candidate.resolve())

    return str(repo_candidate.resolve())


def _sqlite_path_from_jdbc(jdbc_url: Optional[str]) -> Optional[str]:
    if not jdbc_url:
        return None
    prefix = "jdbc:sqlite:"
    if not jdbc_url.startswith(prefix):
        return None
    return jdbc_url[len(prefix) :] or None


def parse_db_connector_settings(node_dir: Optional[Path]) -> DBConnectorSettings:
    if not node_dir:
        return DBConnectorSettings()
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return DBConnectorSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
    model = first_el(root, ".//*[local-name()='config' and @key='model']")
    session = first_el(model, "./*[local-name()='config' and @key='session_info']") if model is not None else None
    auth = first_el(model, "./*[local-name()='config' and @key='authentication']") if model is not None else None

    jdbc_url = first(session, "./*[local-name()='entry' and @key='db_url']/@value") if session is not None else None
    sqlite_path = _sqlite_path_from_jdbc(jdbc_url)
    source = "settings.xml"

    sidecar = _load_sidecar(node_dir)
    if isinstance(sidecar.get("sqlite_path"), str) and sidecar["sqlite_path"].strip():
        sqlite_path = _resolve_sidecar_path(sidecar["sqlite_path"].strip(), node_dir)
        source = "knime2py.local.json"

    return DBConnectorSettings(
        db_type=first(session, "./*[local-name()='entry' and @key='db_type']/@value") if session is not None else None,
        db_dialect=first(session, "./*[local-name()='entry' and @key='db_dialect']/@value") if session is not None else None,
        db_driver=first(session, "./*[local-name()='entry' and @key='db_driver']/@value") if session is not None else None,
        jdbc_url=jdbc_url,
        username=first(auth, "./*[local-name()='entry' and @key='username']/@value") if auth is not None else None,
        auth_type=first(auth, "./*[local-name()='entry' and @key='selectedType']/@value") if auth is not None else None,
        sqlite_path=sqlite_path,
        source=source,
    )


def generate_imports() -> list[str]:
    return []


def _sqlite_sqlalchemy_url(sqlite_path: Optional[str]) -> Optional[str]:
    if not sqlite_path:
        return None
    return f"sqlite:///{sqlite_path}"


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: list[tuple[str, str]],
    out_ports: Optional[list[str]] = None,
) -> list[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_db_connector_settings(ndir)
    dialect = (settings.db_dialect or settings.db_type or "").lower() or None

    descriptor = {
        "kind": "database",
        "dialect": dialect,
        "db_type": settings.db_type,
        "db_driver": settings.db_driver,
        "jdbc_url": settings.jdbc_url,
        "username": settings.username,
        "auth_type": settings.auth_type,
        "sqlite_path": settings.sqlite_path,
        "sqlalchemy_url": _sqlite_sqlalchemy_url(settings.sqlite_path) if dialect == "sqlite" else None,
        "source": settings.source,
    }

    lines = [
        f"# {HUB_URL}",
        f"db_connection = {repr(descriptor)}",
    ]

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = db_connection")
    return lines


def get_name() -> str:
    return "DB Connector"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, [], out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
