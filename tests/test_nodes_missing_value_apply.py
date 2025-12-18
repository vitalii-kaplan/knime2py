"""Unit tests for the Missing Value (Apply) node handler."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import xml.etree.ElementTree as ET
from pandas.testing import assert_frame_equal

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import missing_value_apply  # noqa: E402


def _run_handler(tmp_path: Path, model_obj, df: pd.DataFrame):
    incoming = [
        ("DATA_SRC", SimpleNamespace(source_port="1")),
        ("MODEL_SRC", SimpleNamespace(source_port="2")),
    ]
    outgoing = [("OUT", SimpleNamespace(source_port="1"))]

    imports, body = missing_value_apply.handle(
        missing_value_apply.FACTORY,
        "NODE_APPLY",
        str(tmp_path),
        incoming,
        outgoing,
    )
    code = "\n".join(imports + body)
    env = {
        "context": {
            "DATA_SRC:1": df.copy(),
            "MODEL_SRC:2": model_obj,
        },
        "pd": pd,
        "json": json,
        "ET": ET,
    }
    exec(code, env, env)
    return env["context"]["NODE_APPLY:1"]


def _pmml_from_metadata(metadata: dict) -> str:
    payload = json.dumps(metadata, ensure_ascii=False)
    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<PMML version='4.4' xmlns='http://www.dmg.org/PMML-4_4'>\n"
        "  <Extension name='missing_value_metadata' extender='knime2py'>"
        + payload
        + "</Extension>\n"
        "</PMML>"
    )


def test_missing_value_apply_reads_pmml_metadata(tmp_path: Path):
    df = pd.DataFrame({
        "CabinLetter": ["A", None],
        "Fare": [10.0, None],
    })
    metadata = {
        "column_strategies": [
            {"column": "CabinLetter", "dtype": "string", "strategy": "fixed", "value": "O"},
        ],
        "type_strategies": [
            {"dtype": "float", "strategy": "mean", "value": None},
        ],
    }
    model = _pmml_from_metadata(metadata)
    result = _run_handler(tmp_path, model, df)

    expected = df.copy()
    expected["CabinLetter"] = ["A", "O"]
    expected["Fare"] = [10.0, 10.0]
    assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))


def test_missing_value_apply_legacy_bundle(tmp_path: Path):
    df = pd.DataFrame({"x": [1, None, 3]}, dtype="Int64")
    bundle = {"strategies": [{"dtype": "int", "strategy": "fixed", "value": "0"}]}
    result = _run_handler(tmp_path, bundle, df)

    expected = pd.DataFrame({"x": [1, 0, 3]}, dtype="Int64")
    assert_frame_equal(result, expected)
