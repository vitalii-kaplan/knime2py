from pathlib import Path

import pandas as pd

from knime2py.nodes import joiner


def test_joiner_parses_issue21_settings() -> None:
    cfg = joiner.parse_joiner_settings(Path("tests/data/Kaggle_Titanic_Issue21/Joiner (#1491)"))

    assert cfg.include_matches is True
    assert cfg.include_left_unmatched is False
    assert cfg.include_right_unmatched is False
    assert cfg.suffix == " (Right)"
    assert len(cfg.criteria) == 1
    assert cfg.criteria[0].left_row_id is True
    assert cfg.criteria[0].right_row_id is True


def test_joiner_emitted_code_inner_joins_on_row_id_and_suffixes_right_columns() -> None:
    body = "\n".join(
        joiner.generate_py_body(
            "1491",
            "tests/data/Kaggle_Titanic_Issue21/Joiner (#1491)",
            [("1495", "1"), ("1492", "1")],
            ["1"],
        )
    )

    context = {
        "1495:1": pd.DataFrame(
            {"Name": ["left-0", "left-1", "left-2"], "PassengerId": [10, 11, 12]},
            index=[0, 1, 2],
        ),
        "1492:1": pd.DataFrame(
            {"Name": ["right-1", "right-2", "right-3"], "PassengerId": [21, 22, 23]},
            index=[1, 2, 3],
        ),
    }

    exec(body, {"pd": pd, "context": context})

    out = context["1491:1"]
    assert out["Name"].tolist() == ["left-1", "left-2"]
    assert out["Name (Right)"].tolist() == ["right-1", "right-2"]
    assert out["PassengerId"].tolist() == [11, 12]
    assert out["PassengerId (Right)"].tolist() == [21, 22]
