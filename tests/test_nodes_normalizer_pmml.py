"""Tests for the Normalizer (PMML) handler."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pandas.testing import assert_frame_equal

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import normalizer_pmml, gbt_pmml_exporter  # noqa: E402


PMML_SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig">
  <config key="model">
    <entry key="mode" type="xint" value="2"/>
    <entry key="newmin" type="xdouble" value="0.0"/>
    <entry key="newmax" type="xdouble" value="1.0"/>
    <config key="columns">
      <entry key="0" type="xstring" value="Age"/>
      <entry key="1" type="xstring" value="Fare"/>
    </config>
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


def test_normalizer_pmml_generates_pmml(tmp_path: Path):
    df = pd.DataFrame({"Age": [10.0, 20.0, 30.0], "Fare": [5.0, 10.0, 15.0], "Extra": [1, 2, 3]})
    result_df, pmml = _run_handler(tmp_path, PMML_SETTINGS, df)

    expected = df.copy()
    expected["Age"] = (expected["Age"] - 10.0) / (30.0 - 10.0)
    expected["Fare"] = (expected["Fare"] - 5.0) / (15.0 - 5.0)
    assert_frame_equal(result_df[["Age", "Fare"]], expected[["Age", "Fare"]])

    assert "TransformationDictionary" in pmml
    assert "<DerivedField" in pmml
    assert "<NormContinuous" in pmml


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
