"""Unit tests for the Normalizer Apply (PMML) node handler."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pandas.testing import assert_frame_equal

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import normalizer_pmml_apply  # noqa: E402


def _run_handler(tmp_path: Path, model: dict, df: pd.DataFrame):
    incoming = [
        ("DATA_SRC", SimpleNamespace(source_port="1")),
        ("MODEL_SRC", SimpleNamespace(source_port="2")),
    ]
    outgoing = [("OUT", SimpleNamespace(source_port="1"))]

    imports, body = normalizer_pmml_apply.handle(
        normalizer_pmml_apply.FACTORY,
        "NODE_PMML_APPLY",
        str(tmp_path),
        incoming,
        outgoing,
    )
    code = "\n".join(imports + body)
    env = {
        "context": {
            "DATA_SRC:1": df.copy(),
            "MODEL_SRC:2": model,
        },
        "pd": pd,
    }
    exec(code, env, env)
    return env["context"]["NODE_PMML_APPLY:1"]


def test_normalizer_pmml_apply_minmax(tmp_path: Path):
    df = pd.DataFrame({"x": [0.0, 10.0, 20.0]})
    pmml_bundle = {
        "mode": "MINMAX",
        "new_min": 0.0,
        "new_max": 1.0,
        "columns": ["x"],
        "stats": {"x": {"min": 0.0, "max": 20.0}},
    }
    result = _run_handler(tmp_path, pmml_bundle, df)
    expected = pd.DataFrame({"x": [0.0, 0.5, 1.0]})
    assert_frame_equal(result, expected)


def test_normalizer_pmml_apply_zscore(tmp_path: Path):
    df = pd.DataFrame({"z": [2.0, 4.0, 6.0]})
    pmml_bundle = {
        "mode": "ZSCORE",
        "columns": ["z"],
        "stats": {"z": {"mean": 4.0, "std": 2.0}},
    }
    result = _run_handler(tmp_path, pmml_bundle, df)
    expected = pd.DataFrame({"z": [-1.0, 0.0, 1.0]})
    assert_frame_equal(result, expected)


def test_normalizer_pmml_apply_decimal_scaling(tmp_path: Path):
    df = pd.DataFrame({"d": [100.0, 200.0]})
    pmml_bundle = {
        "mode": "DECIMALSCALING",
        "columns": ["d"],
        "stats": {"d": {"scale": 2}},
    }
    result = _run_handler(tmp_path, pmml_bundle, df)
    expected = pd.DataFrame({"d": [1.0, 2.0]})
    assert_frame_equal(result, expected)
