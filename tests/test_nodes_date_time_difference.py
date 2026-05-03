from pathlib import Path

import pandas as pd

from knime2py.nodes import date_time_difference


def test_date_time_difference_parses_issue23_settings() -> None:
    settings = date_time_difference.parse_date_time_difference_settings(
        Path("tests/data/Example_Date_Issue23/Date_Time Difference (#1501)")
    )

    assert settings.first_column == "as_time"
    assert settings.second_value_type == "COLUMN"
    assert settings.second_column == "uc_created"
    assert settings.mode == "SECOND_MINUS_FIRST"
    assert settings.output_column_name == "Date&Time Difference"


def test_date_time_difference_emitted_code_appends_timedelta_column() -> None:
    body = "\n".join(
        date_time_difference.generate_py_body(
            "1501",
            "tests/data/Example_Date_Issue23/Date_Time Difference (#1501)",
            [("SRC", "1")],
            ["1"],
        )
    )
    context = {
        "SRC:1": pd.DataFrame(
            {
                "uc_created": pd.to_datetime(["2022-02-01 08:53"]),
                "as_time": pd.to_datetime(["2022-12-23 19:55"]),
            }
        )
    }

    exec(body, {"pd": pd, "context": context})

    out = context["1501:1"]
    assert out.loc[0, "Date&Time Difference"] == pd.Timedelta(minutes=-468662)
