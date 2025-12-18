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
    pmml_text = "<PMML version='4.4'></PMML>"

    env = {"context": {"SRC:1": pmml_text}}
    exec(code, env, env)

    settings = pmml_writer.parse_pmml_writer_settings(node_dir)
    assert settings.path is not None
    out_path = Path(settings.path)
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == pmml_text
