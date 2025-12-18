"""Tests for the Model Writer handler."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import model_writer  # noqa: E402


SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig">
  <config key="model">
    <config key="filechooser">
      <config key="path">
        <entry key="file_system_type" type="xstring" value="RELATIVE"/>
        <entry key="file_system_specifier" type="xstring" value="knime.workflow"/>
        <entry key="path" type="xstring" value="../!output/model_output.pkl"/>
      </config>
      <entry key="create_missing_folders" type="xboolean" value="true"/>
    </config>
  </config>
</config>
"""


def test_model_writer_serializes_object(tmp_path: Path):
    workflow = tmp_path / "workflow"
    node_dir = workflow / "Model Writer (#1)"
    node_dir.mkdir(parents=True)
    node_dir.joinpath("settings.xml").write_text(SETTINGS, encoding="utf-8")

    incoming = [("SRC", SimpleNamespace(source_port="1"))]
    outgoing = []

    imports, body = model_writer.handle(
        model_writer.FACTORY,
        "NODE_MODEL_W",
        str(node_dir),
        incoming,
        outgoing,
    )

    code = "\n".join(imports + body)
    model_obj = {"coef": [1, 2, 3], "intercept": 0.5}
    env = {"context": {"SRC:1": model_obj}}
    exec(code, env, env)

    settings = model_writer.parse_model_writer_settings(node_dir)
    assert settings.path is not None
    out_path = Path(settings.path)
    assert out_path.exists()
    loaded = pickle.loads(out_path.read_bytes())
    assert loaded == model_obj
