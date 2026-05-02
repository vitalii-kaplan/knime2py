from pathlib import Path

import pandas as pd

from knime2py.nodes import row_filter


def test_value_filter_legacy_uses_default_values() -> None:
    node_dir = Path("tests/data/Kaggle_Titanic_Issue21/Value Filter _legacy_ (#1495)")

    cfg = row_filter.parse_value_filter_legacy_settings(node_dir)

    assert cfg.column == "Embarked"
    assert cfg.include == ["S", "C"]
    assert cfg.exclude == ["Q"]


def test_value_filter_legacy_emitted_code_filters_input_defaults() -> None:
    node_dir = "tests/data/Kaggle_Titanic_Issue21/Value Filter _legacy_ (#1492)"
    lines = row_filter.generate_value_filter_legacy_py_body(
        "1492",
        node_dir,
        [("2", "1")],
        ["1"],
    )
    body = "\n".join(lines)

    assert "include = ['male']" in body
    assert "exclude = ['female']" in body

    context = {
        "2:1": pd.DataFrame(
            {
                "Sex": ["male", "female", "male", None],
                "PassengerId": [1, 2, 3, 4],
            }
        )
    }
    exec(body, {"pd": pd, "_re": __import__("re"), "context": context})

    out = context["1492:1"]
    assert out["PassengerId"].tolist() == [1, 3]
