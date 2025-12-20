"""Tests for the Normalizer (PMML) handler."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import normalizer_pmml, gbt_pmml_exporter  # noqa: E402


MINMAX_SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.knime.org/2008/09/XMLConfig http://www.knime.org/XMLConfig_2008_09.xsd" key="settings.xml">
  <config key="model">
    <entry key="mode" type="xint" value="1"/>
    <entry key="newmin" type="xdouble" value="0.0"/>
    <entry key="newmax" type="xdouble" value="1.0"/>
    <config key="columns">
      <entry key="array-size" type="xint" value="3"/>
      <entry key="0" type="xstring" value="Age"/>
      <entry key="1" type="xstring" value="Fare"/>
      <entry key="2" type="xstring" value="Pclass"/>
    </config>
    <entry key="all_numeric_columns_used" type="xboolean" value="true"/>
  </config>
</config>
"""

ZSCORE_SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.knime.org/2008/09/XMLConfig http://www.knime.org/XMLConfig_2008_09.xsd" key="settings.xml">
  <config key="model">
    <entry key="mode" type="xint" value="2"/>
    <entry key="newmin" type="xdouble" value="0.0"/>
    <entry key="newmax" type="xdouble" value="1.0"/>
    <config key="columns">
      <entry key="array-size" type="xint" value="3"/>
      <entry key="0" type="xstring" value="Age"/>
      <entry key="1" type="xstring" value="Fare"/>
      <entry key="2" type="xstring" value="Pclass"/>
    </config>
    <entry key="all_numeric_columns_used" type="xboolean" value="true"/>
  </config>
</config>
"""

DECIMAL_SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.knime.org/2008/09/XMLConfig http://www.knime.org/XMLConfig_2008_09.xsd" key="settings.xml">
  <config key="model">
    <entry key="mode" type="xint" value="3"/>
    <entry key="newmin" type="xdouble" value="0.0"/>
    <entry key="newmax" type="xdouble" value="1.0"/>
    <config key="columns">
      <entry key="array-size" type="xint" value="3"/>
      <entry key="0" type="xstring" value="Age"/>
      <entry key="1" type="xstring" value="Fare"/>
      <entry key="2" type="xstring" value="Pclass"/>
    </config>
    <entry key="all_numeric_columns_used" type="xboolean" value="true"/>
  </config>
</config>
"""

def _run_handler(tmp_path: Path, settings_xml: str, df: pd.DataFrame):
    node_dir = tmp_path / "normalizer_pmml"
    node_dir.mkdir()
    node_dir.joinpath("settings.xml").write_text(settings_xml, encoding="utf-8")

    incoming = [("SRC", SimpleNamespace(source_port="1"))]
    outgoing = [("OUT", SimpleNamespace(source_port="1")), ("OUT2", SimpleNamespace(source_port="2"))]

    imports, body = normalizer_pmml.handle(
        normalizer_pmml.FACTORY,
        "NODE_PMML",
        str(node_dir),
        incoming,
        outgoing,
    )
    code = "\n".join(imports + body)
    env = {"context": {"SRC:1": df.copy()}, "pd": pd}
    exec(code, env, env)
    return env["context"]["NODE_PMML:1"], env["context"]["NODE_PMML:2"]


def test_normalizer_pmml_minmax(tmp_path: Path):
    df = pd.DataFrame({"Age": [10.0, 20.0, 30.0], "Fare": [5.0, 10.0, 15.0], "Extra": [1, 2, 3]})
    result_df, pmml = _run_handler(tmp_path, MINMAX_SETTINGS, df)

    expected = df.copy()
    expected["Age"] = (expected["Age"] - 10.0) / (30.0 - 10.0)
    expected["Fare"] = (expected["Fare"] - 5.0) / (15.0 - 5.0)
    assert_frame_equal(result_df[["Age", "Fare"]], expected[["Age", "Fare"]])

    assert isinstance(pmml, dict)
    assert pmml.get("model_type") == "normalizer"
    assert pmml.get("mode") == "MINMAX"
    assert pmml.get("mode") == "MINMAX"
    assert "stats" in pmml

def test_normalizer_pmml_zscore(tmp_path: Path):
    df = pd.DataFrame({"Age": [10.0, 20.0, 30.0], "Fare": [5.0, 10.0, 15.0], "Pclass": [1, 2, 3]})
    result_df, pmml = _run_handler(tmp_path, ZSCORE_SETTINGS, df)

    assert np.allclose(result_df["Age"].tolist(), [-1.22474487, 0.0, 1.22474487], atol=1e-6)
    assert pmml.get("mode") == "ZSCORE"
    stats = pmml.get("stats", {})
    assert "Age" in stats and "mean" in stats["Age"] and "std" in stats["Age"]


def test_normalizer_pmml_decimal_scaling(tmp_path: Path):
    df = pd.DataFrame({"Age": [5.0, 15.0, 25.0], "Fare": [1.0, 2.0, 3.0]})
    result_df, bundle = _run_handler(tmp_path, DECIMAL_SETTINGS, df)

    assert result_df["Age"].abs().max() < 1.0
    assert result_df["Fare"].abs().max() < 1.0
    assert bundle.get("mode") == "DECIMALSCALING"
    stats = bundle.get("stats", {})
    assert stats.get("Age", {}).get("scale") == 2
    assert stats.get("Fare", {}).get("scale") == 1


def test_gbt_pmml_exporter_wraps_bundle(tmp_path: Path):
    bundle = {
        "model": {"dummy": 123},
        "features": ["f1", "f2"],
        "classes": ["A", "B"],
        "n_estimators": 2,
    }
    node_dir = tmp_path / "GBT_PMML"
    node_dir.mkdir()
    node_dir.joinpath("settings.xml").write_text("<?xml version='1.0'?><config/>", encoding="utf-8")

    incoming = [("SRC", SimpleNamespace(source_port="1"))]
    outgoing = [("OUT", SimpleNamespace(source_port="1"))]

    imports, body = gbt_pmml_exporter.handle(
        gbt_pmml_exporter.FACTORY,
        "NODE_GBT_PMML",
        str(node_dir),
        incoming,
        outgoing,
    )
    code = "\n".join(imports + body)
    env = {"context": {"SRC:1": bundle}}
    exec(code, env, env)
    pmml = env["context"]["NODE_GBT_PMML:1"]
    assert "<TreeModel" in pmml
    assert "gbt_metadata" in pmml
    assert "gbt_bundle_json" in pmml
