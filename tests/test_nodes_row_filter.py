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


def test_row_filter_parses_column_v2_and_neq_miss_operator() -> None:
    node_dir = Path("tests/data/INZ_visa_decisions_unification/Row Filter (#1527)")

    cfg = row_filter.parse_row_filter_settings(node_dir)

    assert cfg.match_and is True
    assert cfg.output_mode == "MATCHING"
    assert len(cfg.predicates) == 1
    assert cfg.predicates[0].column == "Country"
    assert cfg.predicates[0].operator == "NEQ_MISS"
    assert cfg.predicates[0].values == ["Total"]


def test_row_filter_emitted_code_removes_inz_total_row() -> None:
    body = "\n".join(
        row_filter.generate_py_body(
            "1527",
            "tests/data/INZ_visa_decisions_unification/Row Filter (#1527)",
            [("1524", "1")],
            ["1"],
        )
    )
    context = {
        "1524:1": pd.DataFrame(
            {
                "Country": ["Afghanistan", "Total", "Zimbabwe"],
                "Number approved": [10, 99, 86],
            }
        )
    }

    exec(body, {"pd": pd, "_re": __import__("re"), "context": context})

    out = context["1527:1"]
    assert out["Country"].tolist() == ["Afghanistan", "Zimbabwe"]
