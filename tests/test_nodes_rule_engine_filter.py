from pathlib import Path

import pandas as pd

from knime2py.nodes import rule_engine_filter


def test_rule_engine_filter_parses_boolean_rules() -> None:
    cfg = rule_engine_filter.parse_rule_engine_filter_settings(
        Path("tests/data/Kaggle_Titanic_Issue21/Rule_based Row Filter (#1494)")
    )

    assert cfg.include is True
    assert len(cfg.rules) == 2
    assert cfg.rules[0].col == "Age"
    assert cfg.rules[0].op == "<"
    assert cfg.rules[0].value == "18"
    assert cfg.rules[0].outcome == "FALSE"
    assert cfg.rules[1].kind == "true"
    assert cfg.rules[1].outcome == "TRUE"


def test_rule_engine_filter_emitted_code_removes_false_rows() -> None:
    body = "\n".join(
        rule_engine_filter.generate_py_body(
            "1494",
            "tests/data/Kaggle_Titanic_Issue21/Rule_based Row Filter (#1494)",
            [("1493", "1")],
            ["1"],
        )
    )
    context = {
        "1493:1": pd.DataFrame(
            {
                "Age": pd.Series([17, 18, 22, pd.NA], dtype="Float64"),
                "PassengerId": [1, 2, 3, 4],
            }
        )
    }

    exec(body, {"pd": pd, "context": context})

    out = context["1494:1"]
    assert out["PassengerId"].tolist() == [2, 3, 4]
