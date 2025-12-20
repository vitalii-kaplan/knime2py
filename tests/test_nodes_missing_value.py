"""Integration-style test for the Missing Value node handler."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pandas.testing import assert_frame_equal

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import missing_value  # noqa: E402


def _run_missing_value(node_dir: Path, input_df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    incoming = [("SRC", SimpleNamespace(source_port="1"))]
    outgoing = [
        ("DST", SimpleNamespace(source_port="1")),
        ("DST", SimpleNamespace(source_port="2")),
    ]
    imports, body = missing_value.handle(
        missing_value.FACTORY,
        "NODE5",
        str(node_dir),
        incoming,
        outgoing,
    )
    code = "\n".join(imports + body)
    env = {
        "context": {"SRC:1": input_df.copy()},
        "pd": pd,
    }
    exec(code, env, env)
    return env["context"]["NODE5:1"], env["context"].get("NODE5:2")


def test_missing_value_matches_expected_output(tmp_path: Path):
    node_dir = repo_root / "tests" / "data" / "Node_missing_value_mode"
    assert node_dir.exists(), "Missing Value settings not found"

    input_path = repo_root / "tests" / "data" / "data" / "Missing_Value" / "imputation_data_input.csv"
    expected_path = repo_root / "tests" / "data" / "data" / "Missing_Value" / "imputation_data_output.csv"
    assert input_path.exists(), "Missing input CSV"
    assert expected_path.exists(), "Missing expected output CSV"

    input_df = pd.read_csv(input_path)
    expected_df = pd.read_csv(expected_path)

    result_df, model_bundle = _run_missing_value(node_dir, input_df)

    # Keep column order and index for stable comparison.
    result_df = result_df[expected_df.columns].reset_index(drop=True)
    expected_df = expected_df.reset_index(drop=True)

    assert_frame_equal(result_df, expected_df, check_dtype=False)
    assert isinstance(model_bundle, dict)
    assert model_bundle.get("model_type") == "missing_value"
    assert "data_dictionary" in model_bundle
    assert "transformations" in model_bundle
