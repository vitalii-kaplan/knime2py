from pathlib import Path

import pandas as pd

from knime2py.nodes import string_to_datetime


def test_string_to_datetime_parses_issue23_settings() -> None:
    settings = string_to_datetime.parse_string_to_datetime_settings(
        Path("tests/data/Example_Date_Issue23/String to Date_Time (#1503)")
    )

    assert settings.columns == ["uc_created", "as_time"]
    assert settings.knime_format == "dd.MM.yyyy HH:mm"
    assert settings.pandas_format == "%d.%m.%Y %H:%M"
    assert settings.temporal_type == "DATE_TIME"
    assert settings.on_error == "SET_MISSING"
    assert settings.append_or_replace == "REPLACE"


def test_string_to_datetime_emitted_code_replaces_selected_columns() -> None:
    body = "\n".join(
        string_to_datetime.generate_py_body(
            "1503",
            "tests/data/Example_Date_Issue23/String to Date_Time (#1503)",
            [("SRC", "1")],
            ["1"],
        )
    )
    context = {
        "SRC:1": pd.DataFrame(
            {
                "uc_created": ["01.02.2022 08:53", "bad"],
                "as_time": ["23.12.2022 19:55", "02.01.2023 00:01"],
                "value": [1, 2],
            }
        )
    }

    exec(body, {"pd": pd, "context": context})

    out = context["1503:1"]
    assert str(out.loc[0, "uc_created"]) == "2022-02-01 08:53:00"
    assert str(out.loc[0, "as_time"]) == "2022-12-23 19:55:00"
    assert pd.isna(out.loc[1, "uc_created"])
    assert out["value"].tolist() == [1, 2]
