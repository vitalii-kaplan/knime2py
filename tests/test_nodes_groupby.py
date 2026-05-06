from pathlib import Path

import pandas as pd

from knime2py.nodes import groupby


def test_groupby_parses_issue21_settings() -> None:
    settings = groupby.parse_groupby_settings(Path("tests/data/Kaggle_Titanic_Issue21/GroupBy (#1490)"))

    assert settings.group_cols == ["Sex", "Embarked"]
    assert settings.aggregations == []
    assert settings.retain_order is False


def test_groupby_emitted_code_distinct_groups_without_aggregations() -> None:
    body = "\n".join(
        groupby.generate_py_body(
            "1490",
            "tests/data/Kaggle_Titanic_Issue21/GroupBy (#1490)",
            [("1494", "1")],
            ["1"],
        )
    )
    context = {
        "1494:1": pd.DataFrame(
            {
                "Sex": ["male", "male", "male", "female"],
                "Embarked": ["S", "C", "S", "Q"],
                "Age": [22, 30, 40, 20],
            }
        )
    }

    exec(body, {"pd": pd, "context": context})

    out = context["1490:1"]
    assert list(out.columns) == ["Sex", "Embarked"]
    assert out.to_dict("records") == [
        {"Sex": "female", "Embarked": "Q"},
        {"Sex": "male", "Embarked": "C"},
        {"Sex": "male", "Embarked": "S"},
    ]


def test_groupby_emitted_code_handles_versioned_knime_sum_methods() -> None:
    body = "\n".join(
        groupby.generate_py_body(
            "1524",
            "tests/data/INZ_visa_decisions_unification/GroupBy (#1524)",
            [("1523", "1")],
            ["1"],
        )
    )
    context = {
        "1523:1": pd.DataFrame(
            {
                "Country": ["A", "A", "B"],
                "Number approved": [1, 3, 5],
                "Number declined": [2, 4, 6],
            }
        )
    }

    exec(body, {"pd": pd, "context": context})

    out = context["1524:1"]
    assert list(out.columns) == ["Country", "Sum(Number approved)", "Sum(Number declined)"]
    assert out.to_dict("records") == [
        {"Country": "A", "Sum(Number approved)": 4, "Sum(Number declined)": 6},
        {"Country": "B", "Sum(Number approved)": 5, "Sum(Number declined)": 6},
    ]
