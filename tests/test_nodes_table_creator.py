from pathlib import Path

import pandas as pd

from knime2py.nodes import table_creator


def test_table_creator_parses_issue23_settings() -> None:
    settings = table_creator.parse_table_creator_settings(
        Path("tests/data/Example_Issue23/Table Creator (#1507)")
    )

    assert [col.name for col in settings.columns] == ["Student", "Course", "Result"]
    assert [col.cell_class.rsplit(".", 1)[-1] for col in settings.columns] == [
        "StringCell",
        "StringCell",
        "IntCell",
    ]
    assert settings.rows == [
        ["Alice", "Math", 90],
        ["Alice", "Physics", 80],
        ["Bob", "Math", 70],
        ["Bob", "Physics", 85],
    ]


def test_table_creator_emitted_code_creates_dataframe_without_input() -> None:
    body = "\n".join(
        table_creator.generate_py_body(
            "1507",
            "tests/data/Example_Issue23/Table Creator (#1507)",
            [],
            ["1"],
        )
    )
    context = {}

    exec(body, {"pd": pd, "context": context})

    out = context["1507:1"]
    assert list(out.columns) == ["Student", "Course", "Result"]
    assert out.to_dict("records") == [
        {"Student": "Alice", "Course": "Math", "Result": 90},
        {"Student": "Alice", "Course": "Physics", "Result": 80},
        {"Student": "Bob", "Course": "Math", "Result": 70},
        {"Student": "Bob", "Course": "Physics", "Result": 85},
    ]
