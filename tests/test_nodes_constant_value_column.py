from pathlib import Path

import pandas as pd

from knime2py.nodes import constant_value_column


def test_constant_value_column_parses_issue21_settings() -> None:
    settings = constant_value_column.parse_constant_value_column_settings(
        Path("tests/data/Kaggle_Titanic_Issue21/Constant Value Column Appender (#1493)")
    )

    assert len(settings) == 1
    setting = settings[0]
    assert setting.mode == "APPEND"
    assert setting.append_name == "New column"
    assert setting.cell_class == "org.knime.core.data.def.StringCell"
    assert setting.value == "new_value"


def test_constant_value_column_emitted_code_appends_constant() -> None:
    body = "\n".join(
        constant_value_column.generate_py_body(
            "1493",
            "tests/data/Kaggle_Titanic_Issue21/Constant Value Column Appender (#1493)",
            [("1491", "1")],
            ["1"],
        )
    )
    context = {"1491:1": pd.DataFrame({"Name": ["a", "b"]})}

    exec(body, {"pd": pd, "context": context})

    out = context["1493:1"]
    assert out["Name"].tolist() == ["a", "b"]
    assert out["New column"].tolist() == ["new_value", "new_value"]
