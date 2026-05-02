from pathlib import Path

from knime2py.nodes import db_connector, postgresql_connector


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


def test_postgresql_connector_parses_connection_settings_without_connecting() -> None:
    node_dir = Path("tests/data/Kaggle_Titanic_Issue22/PostgreSQL Connector (#1497)")

    settings = postgresql_connector.parse_postgresql_connector_settings(node_dir)

    assert settings.db_type == "postgres"
    assert settings.db_dialect == "postgres"
    assert settings.db_driver == "built-in-postgres-42.7.3"
    assert settings.username == "admin"
    assert settings.auth_type == "USER_PWD"
    assert settings.host == "localhost"
    assert settings.port == "5432"
    assert settings.database_name == "k2ptest"


def test_postgresql_connector_emitted_code_publishes_descriptor_without_connecting() -> None:
    body = "\n".join(
        postgresql_connector.generate_py_body(
            "1497",
            "tests/data/Kaggle_Titanic_Issue22/PostgreSQL Connector (#1497)",
            [],
            ["1"],
        )
    )
    context = {}

    exec(body, {"context": context})

    descriptor = context["1497:1"]
    assert descriptor["kind"] == "database"
    assert descriptor["dialect"] == "postgres"
    assert descriptor["host"] == "localhost"
    assert descriptor["port"] == "5432"
    assert descriptor["database_name"] == "k2ptest"
    assert descriptor["sqlalchemy_url"] == "postgresql+psycopg://admin@localhost:5432/k2ptest"
