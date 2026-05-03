from pathlib import Path

import pandas as pd

from knime2py.nodes import duplicate_row_filter, python_script


def test_duplicate_row_filter_parses_issue24_settings() -> None:
    settings = duplicate_row_filter.parse_duplicate_row_filter_settings(
        Path("tests/data/Example_Duplicates_Issue24/Duplicate Row Filter (#1511)")
    )

    assert settings.group_columns == ["uc_course", "uc_loyalty", "uc_welcome"]
    assert settings.remove_duplicates is True
    assert settings.row_selection == "FIRST"
    assert settings.retain_order is True


def test_duplicate_row_filter_emitted_code_keeps_first_row_per_group() -> None:
    body = "\n".join(
        duplicate_row_filter.generate_py_body(
            "1511",
            "tests/data/Example_Duplicates_Issue24/Duplicate Row Filter (#1511)",
            [("SRC", "1")],
            ["1"],
        )
    )
    context = {
        "SRC:1": pd.DataFrame(
            {
                "uc_course": [1, 1, 2],
                "uc_loyalty": [0, 0, 0],
                "uc_welcome": [0, 0, 0],
                "as_result": [10, 20, 30],
            }
        )
    }

    exec(body, {"np": __import__("numpy"), "pd": pd, "context": context})

    assert context["1511:1"].to_dict("records") == [
        {"uc_course": 1, "uc_loyalty": 0, "uc_welcome": 0, "as_result": 10},
        {"uc_course": 2, "uc_loyalty": 0, "uc_welcome": 0, "as_result": 30},
    ]


def test_python_script_decodes_issue24_script() -> None:
    settings = python_script.parse_python_script_settings(
        Path("tests/data/Example_Duplicates_Issue24/Python Script (#1512)")
    )

    assert "df[\"Filled\"] = (~df.isna().any(axis=1)).astype(int)" in settings.script


def test_python_script_emitted_code_runs_knime_io_shim() -> None:
    body = "\n".join(
        python_script.generate_py_body(
            "1512",
            "tests/data/Example_Duplicates_Issue24/Python Script (#1512)",
            [("SRC", "1")],
            ["1"],
        )
    )
    context = {"SRC:1": pd.DataFrame({"a": [1, None], "b": [2, 3]})}

    exec(body, {"pd": pd, "sys": __import__("sys"), "types": __import__("types"), "context": context})

    assert context["1512:1"]["Filled"].tolist() == [1, 0]
