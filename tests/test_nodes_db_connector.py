from pathlib import Path

from knime2py.nodes import db_connector


def test_db_connector_uses_local_sqlite_sidecar() -> None:
    node_dir = Path("tests/data/Kaggle_Titanic_Issue22/DB Connector (#1499)")

    settings = db_connector.parse_db_connector_settings(node_dir)

    assert settings.db_type == "sqlite"
    assert settings.db_dialect == "sqlite"
    assert settings.jdbc_url == "jdbc:sqlite:/Users/vitaly/Home/harbour/KNIME/data/Kaggle_Titanic_Issue22/db.sqlite3"
    assert settings.source == "knime2py.local.json"
    assert settings.sqlite_path is not None
    assert settings.sqlite_path.endswith("tests/data/data/Kaggle_Titanic_Issue22/db.sqlite3")


def test_db_connector_emitted_code_publishes_descriptor() -> None:
    body = "\n".join(
        db_connector.generate_py_body(
            "1499",
            "tests/data/Kaggle_Titanic_Issue22/DB Connector (#1499)",
            [],
            ["1"],
        )
    )
    context = {}

    exec(body, {"context": context})

    descriptor = context["1499:1"]
    assert descriptor["kind"] == "database"
    assert descriptor["dialect"] == "sqlite"
    assert descriptor["sqlite_path"].endswith("tests/data/data/Kaggle_Titanic_Issue22/db.sqlite3")
    assert descriptor["sqlalchemy_url"].startswith("sqlite:////")
    assert descriptor["source"] == "knime2py.local.json"
