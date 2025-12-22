"""Tests for the Model Reader node handler."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import model_reader  # noqa: E402


def _write_settings(node_dir: Path, rel_path: str) -> None:
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig" key="settings.xml">
  <config key="model">
    <config key="filechooser">
      <config key="path">
        <entry key="file_system_type" type="xstring" value="RELATIVE"/>
        <entry key="file_system_specifier" type="xstring" value="knime.workflow"/>
        <entry key="path" type="xstring" value="{rel_path}"/>
      </config>
    </config>
  </config>
</config>
"""
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "settings.xml").write_text(xml, encoding="utf-8")


def test_model_reader_loads_pickle(tmp_path: Path):
    workflow = tmp_path / "workflow"
    node_dir = workflow / "Model Reader (#1)"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {"coef": [1.0, 2.0], "bias": 0.5}
    model_path = data_dir / "model.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(payload, fh)

    _write_settings(node_dir, "../data/model.pkl")

    outgoing = [("OUT", SimpleNamespace(source_port="1"))]
    imports, body = model_reader.handle(
        model_reader.FACTORY,
        "NODE_MODEL_READER",
        str(node_dir),
        incoming=[],
        outgoing=outgoing,
    )
    code = "\n".join(imports + body)
    env = {"context": {}}
    exec(code, env, env)
    assert env["context"]["NODE_MODEL_READER:1"] == payload
