"""Integration test for the Normalizer node exporter."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pandas.testing import assert_frame_equal

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import normalizer  # noqa: E402


def _run_normalizer(settings_src: Path, df: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    node_dir = tmp_path / "normalizer_node"
    node_dir.mkdir()
    shutil.copy(settings_src, node_dir / "settings.xml")

    incoming = [("DATA_SRC", SimpleNamespace(source_port="1"))]
    outgoing = [("OUT", SimpleNamespace(source_port="1"))]
    imports, body = normalizer.handle(
        normalizer.FACTORY,
        "NODE_NORMALIZER",
        str(node_dir),
        incoming,
        outgoing,
    )
    code = "\n".join(imports + body)
    env = {
        "context": {"DATA_SRC:1": df.copy()},
        "pd": pd,
    }
    exec(code, env, env)
    return env["context"]["NODE_NORMALIZER:1"]


def test_normalizer_matches_expected_output(tmp_path: Path):
    settings_src = repo_root / "tests" / "data" / "data" / "Normalizer" / "normalizer_settings.xml"
    input_path = repo_root / "tests" / "data" / "data" / "Normalizer" / "normalizer_data_input.csv"
    expected_path = repo_root / "tests" / "data" / "data" / "Normalizer" / "normalizer_data_output.csv"

    assert settings_src.exists(), "Missing Normalizer settings.xml"
    assert input_path.exists(), "Missing Normalizer input CSV"
    assert expected_path.exists(), "Missing Normalizer expected CSV"

    input_df = pd.read_csv(input_path)
    expected_df = pd.read_csv(expected_path)

    result_df = _run_normalizer(settings_src, input_df, tmp_path)
    #result_df = result_df[expected_df.columns].reset_index(drop=True)
    #expected_df = expected_df.reset_index(drop=True)

    assert_frame_equal(result_df, expected_df, check_dtype=False)
