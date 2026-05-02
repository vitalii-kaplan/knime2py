from pathlib import Path

import pandas as pd
import sqlite3

from knime2py.nodes import db_connector, db_query_reader


def test_db_query_reader_parses_sql_statement() -> None:
    settings = db_query_reader.parse_db_query_reader_settings(
        Path("tests/data/Kaggle_Titanic_Issue22/DB Query Reader (#1500)")
    )

    assert settings.sql_statement == "SELECT * from jobs_jobsettingsmeta"


def test_db_query_reader_emitted_code_reads_sqlite_descriptor() -> None:
    connector = db_connector.parse_db_connector_settings(
        Path("tests/data/Kaggle_Titanic_Issue22/DB Connector (#1499)")
    )
    context = {
        "1499:1": {
            "kind": "database",
            "dialect": "sqlite",
            "sqlite_path": connector.sqlite_path,
        }
    }
    body = "\n".join(
        db_query_reader.generate_py_body(
            "1500",
            "tests/data/Kaggle_Titanic_Issue22/DB Query Reader (#1500)",
            [("1499", "1")],
            ["1"],
        )
    )

    exec(body, {"pd": pd, "sqlite3": sqlite3, "context": context})

    out = context["1500:1"]
    assert list(out.columns) == ["id", "created_at", "file_name", "factory", "node_name", "name", "job_status"]
    assert len(out) == 141
