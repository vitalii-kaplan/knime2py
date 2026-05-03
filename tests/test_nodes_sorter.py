from pathlib import Path
import re

import pandas as pd

from knime2py.nodes import sorter


def test_sorter_parses_issue23_settings() -> None:
    settings = sorter.parse_sorter_settings(Path("tests/data/Example_Date_Issue23/Sorter (#1500)"))

    assert [(criterion.column, criterion.ascending) for criterion in settings.criteria] == [
        ("Date&Time Difference", False)
    ]
    assert settings.missing_to_end is False


def test_sorter_emitted_code_sorts_existing_columns() -> None:
    body = "\n".join(
        sorter.generate_py_body(
            "1500",
            "tests/data/Example_Date_Issue23/Sorter (#1500)",
            [("SRC", "1")],
            ["1"],
        )
    )
    context = {"SRC:1": pd.DataFrame({"Date&Time Difference": [1, 3, 2], "value": ["a", "c", "b"]})}

    exec(body, {"pd": pd, "re": re, "context": context})

    assert context["1500:1"]["Date&Time Difference"].tolist() == [3, 2, 1]


def test_sorter_emitted_code_passes_through_when_columns_are_stale() -> None:
    body = "\n".join(
        sorter.generate_py_body(
            "1500",
            "tests/data/Example_Date_Issue23/Sorter (#1500)",
            [("SRC", "1")],
            ["1"],
        )
    )
    context = {"SRC:1": pd.DataFrame({"other": [2, 1]})}

    exec(body, {"pd": pd, "re": re, "context": context})

    assert context["1500:1"].to_dict("records") == [{"other": 2}, {"other": 1}]


def test_sorter_emitted_code_matches_knime_duration_descending_order() -> None:
    body = "\n".join(
        sorter.generate_py_body(
            "1500",
            "tests/data/Example_Date_Issue23/Sorter (#1500)",
            [("SRC", "1")],
            ["1"],
        )
    )
    context = {
        "SRC:1": pd.DataFrame(
            {
                "Date&Time Difference": pd.to_timedelta(
                    ["0 days 01:00:00", "-1 days +00:00:00", "0 days 00:00:00", pd.NaT, "0 days 02:00:00"]
                )
            }
        )
    }

    exec(body, {"pd": pd, "re": re, "context": context})

    assert context["1500:1"]["Date&Time Difference"].tolist()[:4] == [
        pd.Timedelta(days=-1),
        pd.Timedelta(hours=2),
        pd.Timedelta(hours=1),
        pd.Timedelta(0),
    ]
