"""Integration test for the Normalizer node exporter."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import normalizer  # noqa: E402


def _run_normalizer(settings_src: Path, df: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    result_df, _ = _run_normalizer_from_string(settings_src.read_text(encoding="utf-8"), df, tmp_path)
    return result_df


def _run_normalizer_from_string(settings_xml: str, df: pd.DataFrame, tmp_path: Path) -> tuple[pd.DataFrame, object]:
    node_dir = tmp_path / ("normalizer_node_" + str(abs(hash(settings_xml)) % 10000))
    node_dir.mkdir()
    node_dir.joinpath("settings.xml").write_text(settings_xml, encoding="utf-8")

    incoming = [("DATA_SRC", SimpleNamespace(source_port="1"))]
    outgoing = [("OUT", SimpleNamespace(source_port="1")), ("MODEL", SimpleNamespace(source_port="2"))]
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
    return env["context"]["NODE_NORMALIZER:1"], env["context"].get("NODE_NORMALIZER:2")


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

    assert_frame_equal(result_df, expected_df, check_dtype=False)


MINMAX_SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig" key="settings.xml">
  <config key="model">
    <config key="dataColumnFilterConfig">
      <entry key="mode" type="xstring" value="MANUAL"/>
      <config key="patternFilter">
        <entry key="pattern" type="xstring" value=""/>
        <entry key="isCaseSensitive" type="xboolean" value="false"/>
        <entry key="isInverted" type="xboolean" value="false"/>
      </config>
      <config key="manualFilter">
        <config key="manuallySelected">
          <entry key="array-size" type="xint" value="0"/>
        </config>
        <config key="manuallyDeselected">
          <entry key="array-size" type="xint" value="0"/>
        </config>
      </config>
    </config>
    <entry key="mode" type="xstring" value="MINMAX"/>
    <entry key="new-min" type="xdouble" value="0.0"/>
    <entry key="new-max" type="xdouble" value="1.0"/>
  </config>
</config>
"""

ZSCORE_SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig" key="settings.xml">
  <config key="model">
    <config key="dataColumnFilterConfig">
      <entry key="mode" type="xstring" value="MANUAL"/>
      <config key="manualFilter">
        <config key="manuallySelected">
          <entry key="array-size" type="xint" value="0"/>
        </config>
        <config key="manuallyDeselected">
          <entry key="array-size" type="xint" value="0"/>
        </config>
      </config>
    </config>
    <entry key="mode" type="xstring" value="Z_SCORE"/>
    <entry key="new-min" type="xdouble" value="0.0"/>
    <entry key="new-max" type="xdouble" value="1.0"/>
  </config>
</config>
"""

DECIMAL_SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig" key="settings.xml">
  <config key="model">
    <config key="dataColumnFilterConfig">
      <entry key="mode" type="xstring" value="MANUAL"/>
      <config key="manualFilter">
        <config key="manuallySelected">
          <entry key="array-size" type="xint" value="0"/>
        </config>
        <config key="manuallyDeselected">
          <entry key="array-size" type="xint" value="0"/>
        </config>
      </config>
    </config>
    <entry key="mode" type="xstring" value="DECIMALSCALING"/>
    <entry key="new-min" type="xdouble" value="0.0"/>
    <entry key="new-max" type="xdouble" value="1.0"/>
  </config>
</config>
"""


def test_normalizer_minmax_mode(tmp_path: Path):
    df = pd.DataFrame({"Age": [10.0, 20.0, 30.0], "Fare": [5.0, 10.0, 15.0]})
    result_df, bundle = _run_normalizer_from_string(MINMAX_SETTINGS, df, tmp_path)
    assert result_df["Age"].between(0, 1).all()
    assert bundle and bundle.get("mode") == "MINMAX"


def test_normalizer_zscore_mode(tmp_path: Path):
    df = pd.DataFrame({"Age": [10.0, 20.0, 30.0]})
    result_df, bundle = _run_normalizer_from_string(ZSCORE_SETTINGS, df, tmp_path)
    expected = np.array([-1.22474487, 0.0, 1.22474487])
    assert np.allclose(result_df["Age"].to_numpy(), expected, atol=1e-6)
    assert bundle and bundle.get("mode") == "ZSCORE"


def test_normalizer_decimal_mode(tmp_path: Path):
    df = pd.DataFrame({"Age": [5.0, 15.0, 25.0]})
    result_df, bundle = _run_normalizer_from_string(DECIMAL_SETTINGS, df, tmp_path)
    assert result_df["Age"].abs().max() < 1.0
    assert bundle and bundle.get("mode") == "DECIMALSCALING"
