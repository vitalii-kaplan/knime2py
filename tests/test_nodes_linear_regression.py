from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from knime2py.nodes.linear_regression_learner import generate_py_body as generate_learner_body
from knime2py.nodes.linear_regression_learner import parse_linear_regression_settings
from knime2py.nodes.regression_predictor import (
    generate_py_body as generate_predictor_body,
    parse_regression_predictor_settings,
)


FIXTURE_ROOT = Path(__file__).resolve().parent / "data" / "INZ_visa_decisions_model"


def test_linear_regression_learner_parses_inz_settings() -> None:
    settings = parse_linear_regression_settings(FIXTURE_ROOT / "Linear Regression Learner (#1576)")

    assert settings.target == "Approval rate"
    assert settings.include_constant is True
    assert settings.missing_value_handling == "fail"
    assert "WB_FY2026_Income_Group" in settings.include_cols
    assert "Country" in settings.exclude_cols


def test_regression_predictor_parses_inz_settings() -> None:
    settings = parse_regression_predictor_settings(FIXTURE_ROOT / "Regression Predictor (#1574)")

    assert settings.has_custom_name is False
    assert settings.custom_name is None


def test_linear_regression_learner_emits_ols_bundle_and_coefficients(tmp_path: Path) -> None:
    node_dir = tmp_path / "Linear Regression Learner"
    node_dir.mkdir()
    (node_dir / "settings.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<config key="settings.xml">
  <config key="model">
    <entry key="target" type="xstring" value="target"/>
    <config key="column_filter">
      <config key="included_names">
        <entry key="array-size" type="xint" value="2"/>
        <entry key="0" type="xstring" value="x"/>
        <entry key="1" type="xstring" value="cat"/>
      </config>
      <config key="excluded_names">
        <entry key="array-size" type="xint" value="0"/>
      </config>
    </config>
    <entry key="include_constant" type="xboolean" value="true"/>
    <entry key="missing_value_handling" type="xstring" value="fail"/>
  </config>
</config>
""",
        encoding="utf-8",
    )
    df = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "cat": ["B", "A", "B", "A"],
        }
    )
    context = {"SRC:1": df}
    body = generate_learner_body(
        "LR",
        str(node_dir),
        [("SRC", "1")],
        ["1", "2"],
    )
    exec(
        "\n".join(body),
        {"context": context, "pd": pd, "np": np, "_scipy_stats": _scipy_stats, "__name__": "__main__"},
    )

    bundle = context["LR:1"]
    coef_df = context["LR:2"]

    assert bundle["kind"] == "linear_regression"
    assert bundle["target"] == "target"
    assert "x" in bundle["features"]
    assert list(coef_df.columns) == ["Variable", "Coeff.", "Std. Err.", "t-value", "P>|t|"]


def test_regression_predictor_scores_linear_regression_bundle() -> None:
    context = {
        "MODEL:1": {
            "kind": "linear_regression",
            "target": "y",
            "features": ["x", "cat=B"],
            "feature_info": [
                {"kind": "numeric", "column": "x", "features": ["x"]},
                {
                    "kind": "categorical",
                    "column": "cat",
                    "levels": ["A", "B"],
                    "drop_level": "A",
                    "features": ["cat=B"],
                },
            ],
            "include_constant": True,
            "coef": [2.0, 10.0, 1.0],
        },
        "DATA:1": pd.DataFrame({"x": [1.0, 2.0], "cat": ["A", "B"]}),
    }
    body = generate_predictor_body(
        "PRED",
        str(FIXTURE_ROOT / "Regression Predictor (#1574)"),
        [("MODEL", "1"), ("DATA", "1")],
        ["1"],
    )

    exec("\n".join(body), {"context": context, "pd": pd, "np": np})

    out = context["PRED:1"]
    assert out["Prediction (y)"].tolist() == [3.0, 15.0]
