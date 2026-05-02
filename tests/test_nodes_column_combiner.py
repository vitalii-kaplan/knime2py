from pathlib import Path

import pandas as pd

from knime2py.nodes import column_combiner


def test_column_combiner_parses_issue22_settings() -> None:
    settings = column_combiner.parse_column_combiner_settings(
        Path("tests/data/Kaggle_Titanic_Issue22/Column Combiner (#1501)")
    )

    assert settings.columns == ["factory", "node_name"]
    assert settings.delimiter == ","
    assert settings.new_column_name == "Combined String"
    assert settings.remove_included_columns is False


def test_column_combiner_emitted_code_combines_selected_columns() -> None:
    body = "\n".join(
        column_combiner.generate_py_body(
            "1501",
            "tests/data/Kaggle_Titanic_Issue22/Column Combiner (#1501)",
            [("1500", "1")],
            ["1"],
        )
    )
    context = {
        "1500:1": pd.DataFrame(
            {
                "factory": ["org.knime.Reader", "org.knime.Writer"],
                "node_name": ["CSV Reader", "CSV Writer"],
                "job_status": ["RUNNING", "SUCCEEDED"],
            }
        )
    }

    exec(body, {"pd": pd, "context": context})

    out = context["1501:1"]
    assert list(out.columns) == ["factory", "node_name", "job_status", "Combined String"]
    assert out["Combined String"].tolist() == [
        "org.knime.Reader,CSV Reader",
        "org.knime.Writer,CSV Writer",
    ]


def test_column_combiner_quotes_only_when_needed() -> None:
    body = "\n".join(
        column_combiner.generate_py_body(
            "1501",
            "tests/data/Kaggle_Titanic_Issue22/Column Combiner (#1501)",
            [("1500", "1")],
            ["1"],
        )
    )
    context = {
        "1500:1": pd.DataFrame(
            {
                "factory": ["org.knime.Reader,legacy"],
                "node_name": ['CSV "Reader"'],
            }
        )
    }

    exec(body, {"pd": pd, "context": context})

    assert context["1501:1"]["Combined String"].tolist() == [
        '"org.knime.Reader,legacy","CSV ""Reader"""'
    ]
