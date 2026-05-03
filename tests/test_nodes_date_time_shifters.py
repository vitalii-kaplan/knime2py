from pathlib import Path
import re

import pandas as pd

from knime2py.nodes import cell_splitter, date_shifter, time_shifter


def test_time_shifter_parses_issue24_settings() -> None:
    settings = time_shifter.parse_time_shifter_settings(Path("tests/data/Example_Issue24/Time Shifter (#1511)"))

    assert settings.columns == ["uc_created"]
    assert settings.shift_mode == "SHIFT_VALUE"
    assert settings.shift_duration_value == "PT0H15M0.000S"
    assert settings.replace_or_append == "REPLACE"


def test_time_shifter_emitted_code_replaces_datetime_column() -> None:
    body = "\n".join(
        time_shifter.generate_py_body(
            "1511",
            "tests/data/Example_Issue24/Time Shifter (#1511)",
            [("SRC", "1")],
            ["1"],
        )
    )
    context = {
        "SRC:1": pd.DataFrame(
            {"uc_created": pd.to_datetime(["2022-06-03 12:19"]), "as_time": pd.to_datetime(["2022-06-03 12:12"])}
        )
    }

    exec(body, {"pd": pd, "re": re, "context": context})

    assert context["1511:1"].loc[0, "uc_created"] == pd.Timestamp("2022-06-03 12:34")


def test_date_shifter_parses_issue24_settings() -> None:
    settings = date_shifter.parse_date_shifter_settings(Path("tests/data/Example_Issue24/Date Shifter (#1512)"))

    assert settings.columns == ["as_time"]
    assert settings.shift_mode == "SHIFT_VALUE"
    assert settings.shift_period_value == "P0Y0M0W1D"
    assert settings.replace_or_append == "REPLACE"


def test_date_shifter_emitted_code_replaces_datetime_column() -> None:
    body = "\n".join(
        date_shifter.generate_py_body(
            "1512",
            "tests/data/Example_Issue24/Date Shifter (#1512)",
            [("SRC", "1")],
            ["1"],
        )
    )
    context = {
        "SRC:1": pd.DataFrame(
            {"uc_created": pd.to_datetime(["2022-06-03 12:34"]), "as_time": pd.to_datetime(["2022-06-03 12:12"])}
        )
    }

    exec(body, {"pd": pd, "re": re, "context": context})

    assert context["1512:1"].loc[0, "as_time"] == pd.Timestamp("2022-06-04 12:12")


def test_cell_splitter_emitted_code_matches_issue24_shape_and_types() -> None:
    body = "\n".join(
        cell_splitter.generate_py_body(
            "1513",
            "tests/data/Example_Issue24/Cell Splitter (#1513)",
            [("SRC", "1")],
            ["1"],
        )
    )
    context = {
        "SRC:1": pd.DataFrame(
            {"as_time": pd.to_datetime(["2022-06-04 12:12"]), "uc_created": pd.to_datetime(["2022-06-03 12:08"])}
        )
    }

    exec(body, {"pd": pd, "context": context})

    out = context["1513:1"]
    assert out.columns.tolist() == ["as_time", "uc_created_Arr[0]", "uc_created_Arr[1]"]
    assert out.loc[0, "uc_created_Arr[0]"] == "2022-06-03T12"
    assert out.loc[0, "uc_created_Arr[1]"] == 8
