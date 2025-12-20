"""Tests for the PMML Writer handler."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import pmml_writer  # noqa: E402


SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig">
  <config key="model">
    <entry key="validate_PMML" type="xboolean" value="true"/>
    <config key="filechooser">
      <config key="path">
        <entry key="file_system_type" type="xstring" value="RELATIVE"/>
        <entry key="file_system_specifier" type="xstring" value="knime.workflow"/>
        <entry key="path" type="xstring" value="../!output/Imputation_output.pmml"/>
      </config>
      <entry key="create_missing_folders" type="xboolean" value="true"/>
    </config>
  </config>
</config>
"""


def _build_env(code: str, payload, out_path: Path):
    env = {"context": {"SRC:1": payload}}
    exec(code, env, env)
    assert out_path.exists()
    return out_path.read_text(encoding="utf-8")


def test_pmml_writer_exports_file(tmp_path: Path):
    workflow = tmp_path / "workflow"
    node_dir = workflow / "PMML Writer (#1)"
    node_dir.mkdir(parents=True)
    node_dir.joinpath("settings.xml").write_text(SETTINGS, encoding="utf-8")

    incoming = [("SRC", SimpleNamespace(source_port="1"))]
    outgoing = []

    imports, body = pmml_writer.handle(
        pmml_writer.FACTORY,
        "NODE_W",
        str(node_dir),
        incoming,
        outgoing,
    )

    code = "\n".join(imports + body)
    settings = pmml_writer.parse_pmml_writer_settings(node_dir)
    assert settings.path is not None
    out_path = Path(settings.path)

    pmml_text = "<PMML version='4.4'></PMML>"
    written = _build_env(code, pmml_text, out_path)
    assert written == pmml_text

    bundle = {
        "model_type": "missing_value",
        "version": "4.2",
        "application": {"name": "knime2py", "version": "1.0"},
        "data_dictionary": [
            {"name": "Age", "optype": "continuous", "dataType": "double", "interval": [0.0, 80.0]},
            {"name": "CabinLetter", "optype": "categorical", "dataType": "string", "values": ["A", "B"]},
        ],
        "transformations": [
            {"column": "Age", "derived_name": "Age*", "const": "29.7", "const_dtype": "double", "optype": "continuous"},
        ],
        "strategies": [],
        "column_strategies": [],
    }
    written_bundle = _build_env(code, bundle, out_path)
    assert "<DataDictionary" in written_bundle
    assert "<TransformationDictionary" in written_bundle

    norm_bundle = {
        "model_type": "normalizer",
        "version": "4.2",
        "application": {"name": "knime2py", "version": "1.0"},
        "mode": "MINMAX",
        "new_min": 0.0,
        "new_max": 1.0,
        "columns": ["Age"],
        "stats": {"Age": {"min": 10.0, "max": 30.0}},
    }
    written_norm = _build_env(code, norm_bundle, out_path)
    assert "<DataDictionary" in written_norm
    assert "<NormContinuous" in written_norm

    decimal_bundle = {
        "model_type": "normalizer",
        "version": "4.2",
        "application": {"name": "knime2py", "version": "1.0"},
        "mode": "DECIMALSCALING",
        "columns": ["Fare"],
        "stats": {"Fare": {"scale": 2}},
    }
    written_dec = _build_env(code, decimal_bundle, out_path)
    assert "Fare_dec" in written_dec
    assert "<Apply function=\"/\"" in written_dec
