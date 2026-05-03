from pathlib import Path

import pandas as pd

from knime2py.nodes import pivot, unpivot


def test_pivot_parses_issue23_settings() -> None:
    settings = pivot.parse_pivot_settings(Path("tests/data/Example_Issue23/Pivot (#1497)"))

    assert settings.group_cols == ["Student"]
    assert settings.pivot_cols == ["Course"]
    assert [(agg.column, agg.method) for agg in settings.aggregations] == [
        ("Result", "Mean_V4.6"),
    ]
    assert settings.missing_values is True


def test_pivot_emitted_code_aggregates_duplicate_group_pivot_rows() -> None:
    body = "\n".join(
        pivot.generate_py_body(
            "1497",
            "tests/data/Example_Issue23/Pivot (#1497)",
            [("SRC", "1")],
            ["1", "2", "3"],
        )
    )
    context = {
        "SRC:1": pd.DataFrame(
            {
                "Student": ["Alice", "Alice", "Alice", "Bob"],
                "Course": ["Math", "Math", "Physics", "Math"],
                "Result": [80, 100, 70, 60],
            }
        )
    }

    exec(body, {"pd": pd, "context": context})

    out = context["1497:1"].sort_values("Student").reset_index(drop=True)
    assert "Math+Mean(Result)" in out.columns
    assert "Physics+Mean(Result)" in out.columns
    assert out.loc[0, "Math+Mean(Result)"] == 90
    assert out.loc[0, "Physics+Mean(Result)"] == 70
    assert context["1497:2"].to_dict("records")[0]["Mean(Result)"] == 83.33333333333333
    assert "Math+Mean(Result)" in context["1497:3"].columns


def test_pivot_emitted_code_handles_missing_nullable_integer_pivot_values() -> None:
    body = "\n".join(
        pivot.generate_py_body(
            "1497",
            "tests/data/Example_Issue23/Pivot (#1497)",
            [("SRC", "1")],
            ["1"],
        )
    )
    context = {
        "SRC:1": pd.DataFrame(
            {
                "Student": ["Alice", "Alice"],
                "Course": pd.Series([1, pd.NA], dtype="Int64"),
                "Result": [10, 20],
            }
        )
    }

    exec(body, {"pd": pd, "context": context})

    out = context["1497:1"]
    assert "1+Mean(Result)" in out.columns
    assert "Missing+Mean(Result)" in out.columns
    assert out.loc[0, "Missing+Mean(Result)"] == 20


def test_unpivot_parses_issue23_settings() -> None:
    settings = unpivot.parse_unpivot_settings(Path("tests/data/Example_Issue23/Unpivot (#1498)"))

    assert settings.value_cols == ["Math+Mean(Result)", "Physics+Mean(Result)"]
    assert "Student" in settings.retained_cols
    assert settings.missing_values is True


def test_unpivot_emitted_code_melts_selected_value_columns() -> None:
    body = "\n".join(
        unpivot.generate_py_body(
            "1498",
            "tests/data/Example_Issue23/Unpivot (#1498)",
            [("SRC", "1")],
            ["1"],
        )
    )
    context = {
        "SRC:1": pd.DataFrame(
            {
                "Student": ["Alice", "Bob"],
                "Math+Mean(Result)": [90.0, 70.0],
                "Physics+Mean(Result)": [80.0, 85.0],
            }
        )
    }

    exec(body, {"pd": pd, "context": context})

    out = context["1498:1"]
    assert list(out.columns) == ["RowIDs", "ColumnNames", "ColumnValues", "Student"]
    assert out.to_dict("records") == [
        {"RowIDs": "Row0", "ColumnNames": "Math+Mean(Result)", "ColumnValues": 90.0, "Student": "Alice"},
        {"RowIDs": "Row0", "ColumnNames": "Physics+Mean(Result)", "ColumnValues": 80.0, "Student": "Alice"},
        {"RowIDs": "Row1", "ColumnNames": "Math+Mean(Result)", "ColumnValues": 70.0, "Student": "Bob"},
        {"RowIDs": "Row1", "ColumnNames": "Physics+Mean(Result)", "ColumnValues": 85.0, "Student": "Bob"},
    ]
