"""Tests for the PMML Reader node handler."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import pmml_reader  # noqa: E402


def _write_settings(node_dir: Path, relative_path: str) -> None:
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig" key="settings.xml">
  <config key="model">
    <config key="filechooser">
      <config key="path">
        <entry key="file_system_type" type="xstring" value="RELATIVE"/>
        <entry key="file_system_specifier" type="xstring" value="knime.workflow"/>
        <entry key="path" type="xstring" value="{relative_path}"/>
      </config>
    </config>
  </config>
</config>
"""
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "settings.xml").write_text(xml, encoding="utf-8")


def test_pmml_reader_loads_text(tmp_path: Path):
    workflow = tmp_path / "workflow"
    node_dir = workflow / "PMML Reader (#1)"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pmml_path = data_dir / "model.pmml"
    pmml_content = "<PMML version='4.2'><Header/></PMML>"
    pmml_path.write_text(pmml_content, encoding="utf-8")

    _write_settings(node_dir, "../data/model.pmml")

    outgoing = [("OUT", SimpleNamespace(source_port="1"))]
    imports, body = pmml_reader.handle(
        pmml_reader.FACTORY,
        "NODE_PMML_READER",
        str(node_dir),
        incoming=[],
        outgoing=outgoing,
    )
    code = "\n".join(imports + body)
    env = {"context": {}}
    with pytest.raises(ValueError):
        exec(code, env, env)


def test_pmml_reader_extracts_strategies_from_pmml(tmp_path: Path):
    workflow = tmp_path / "wf"
    node_dir = workflow / "PMML Reader (#2)"
    data_dir = tmp_path / "mv_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pmml_text = """
<PMML version="4.2" xmlns="http://www.dmg.org/PMML-4_2">
  <Header/>
  <DataDictionary numberOfFields="1">
    <DataField name="Age" optype="continuous" dataType="double"/>
  </DataDictionary>
  <TransformationDictionary>
    <DerivedField name="Age*" displayName="Age" optype="continuous" dataType="double">
      <Apply function="if">
        <Apply function="isMissing">
          <FieldRef field="Age"/>
        </Apply>
        <Constant dataType="double">33.0</Constant>
        <FieldRef field="Age"/>
      </Apply>
    </DerivedField>
  </TransformationDictionary>
</PMML>
""".strip()
    pmml_path = data_dir / "mv.pmml"
    pmml_path.write_text(pmml_text, encoding="utf-8")

    _write_settings(node_dir, "../mv_data/mv.pmml")

    outgoing = [("OUT", SimpleNamespace(source_port="1"))]
    imports, body = pmml_reader.handle(
        pmml_reader.FACTORY,
        "NODE_PMML_READER2",
        str(node_dir),
        incoming=[],
        outgoing=outgoing,
    )
    code = "\n".join(imports + body)
    env = {"context": {}}
    exec(code, env, env)
    assert env["context"]["NODE_PMML_READER2:1"] == {
        "strategies": [],
        "column_strategies": [{"column": "Age", "dtype": "float", "strategy": "fixed", "value": "33.0"}],
    }


def test_pmml_reader_detects_normalizer_bundle(tmp_path: Path):
    workflow = tmp_path / "wf_norm"
    node_dir = workflow / "PMML Reader (#3)"
    data_dir = tmp_path / "norm_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pmml_text = """
<PMML version="4.2" xmlns="http://www.dmg.org/PMML-4_2">
  <Header>
    <Application name="knime2py" version="1.0"/>
  </Header>
  <DataDictionary numberOfFields="1">
    <DataField name="Age" optype="continuous" dataType="double"/>
  </DataDictionary>
  <TransformationDictionary>
    <DerivedField name="Age_norm" optype="continuous" dataType="double">
      <NormContinuous field="Age">
        <LinearNorm orig="0.0" norm="0.0"/>
        <LinearNorm orig="10.0" norm="1.0"/>
      </NormContinuous>
    </DerivedField>
  </TransformationDictionary>
</PMML>
""".strip()
    pmml_path = data_dir / "norm.pmml"
    pmml_path.write_text(pmml_text, encoding="utf-8")

    _write_settings(node_dir, "../norm_data/norm.pmml")

    outgoing = [("OUT", SimpleNamespace(source_port="1"))]
    imports, body = pmml_reader.handle(
        pmml_reader.FACTORY,
        "NODE_PMML_READER3",
        str(node_dir),
        incoming=[],
        outgoing=outgoing,
    )
    code = "\n".join(imports + body)
    env = {"context": {}}
    exec(code, env, env)
    bundle = env["context"]["NODE_PMML_READER3:1"]
    assert bundle["model_type"] == "normalizer"
    assert bundle["mode"] == "MINMAX"
    assert bundle["columns"] == ["Age"]
    assert bundle["stats"]["Age"] == {"min": 0.0, "max": 10.0}
