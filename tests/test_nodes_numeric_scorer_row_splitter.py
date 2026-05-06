from pathlib import Path

import numpy as np
import pandas as pd

from knime2py.nodes.numeric_scorer import generate_py_body as generate_numeric_scorer_body
from knime2py.nodes.numeric_scorer import parse_numeric_scorer_settings
from knime2py.nodes.row_splitter import generate_py_body as generate_row_splitter_body
from knime2py.nodes.row_filter import parse_row_filter_settings


FIXTURE_ROOT = Path(__file__).resolve().parent / "data" / "INZ_visa_decisions_model"


def test_numeric_scorer_parses_inz_settings() -> None:
    settings = parse_numeric_scorer_settings(FIXTURE_ROOT / "Numeric Scorer (#1577)")

    assert settings.reference_col == "Approval rate"
    assert settings.predicted_col == "Prediction (Approval rate)"
    assert settings.override_output_name is False
    assert settings.number_of_predictors == 0


def test_row_splitter_parses_inz_settings() -> None:
    settings = parse_row_filter_settings(FIXTURE_ROOT / "Row Splitter (#1581)")

    assert settings.output_mode == "MATCHING"
    assert settings.predicates[0].column == "Approval rate"
    assert settings.predicates[0].operator == "LT"
    assert settings.predicates[0].values == ["0.9"]


def test_numeric_scorer_emits_expected_metric_order() -> None:
    df = pd.DataFrame(
        {
            "Actual": [1.0, 2.0, 3.0],
            "Pred": [1.0, 2.5, 2.5],
        }
    )
    context = {"SRC:1": df}
    body = generate_numeric_scorer_body("NS", None, [("SRC", "1")], ["1"])
    body = [
        line.replace("_reference_col = 'target'", "_reference_col = 'Actual'")
        .replace("_predicted_col = 'prediction'", "_predicted_col = 'Pred'")
        .replace("_score_col = 'prediction'", "_score_col = 'Pred'")
        for line in body
    ]

    exec("\n".join(body), {"context": context, "pd": pd, "np": np})

    out = context["NS:1"]
    assert list(out.columns) == ["Pred"]
    assert np.isclose(out["Pred"].iloc[0], 0.75)
    assert np.isclose(out["Pred"].iloc[1], 1.0 / 3.0)
    assert np.isclose(out["Pred"].iloc[2], 1.0 / 6.0)
    assert np.isclose(out["Pred"].iloc[3], np.sqrt(1.0 / 6.0))


def test_row_splitter_outputs_matching_and_non_matching_ports() -> None:
    df = pd.DataFrame({"Approval rate": [0.1, 0.9, 1.0], "Country": ["A", "B", "C"]})
    context = {"SRC:1": df}
    body = generate_row_splitter_body(
        "RS",
        str(FIXTURE_ROOT / "Row Splitter (#1581)"),
        [("SRC", "1")],
        ["1", "2"],
    )

    exec("\n".join(body), {"context": context, "pd": pd, "_re": __import__("re")})

    assert context["RS:1"]["Country"].tolist() == ["A"]
    assert context["RS:2"]["Country"].tolist() == ["B", "C"]
