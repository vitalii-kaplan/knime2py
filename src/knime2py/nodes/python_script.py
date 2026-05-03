#!/usr/bin/env python3
from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, first, normalize_in_ports, split_out_imports


FACTORY = "org.knime.python3.scripting.nodes2.script.PythonScriptNodeFactory"

HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.python3.scripting/latest/"
    "org.knime.python3.scripting.nodes2.script.PythonScriptNodeFactory"
)


@dataclass
class PythonScriptSettings:
    script: str = ""


def _decode_script(raw: str) -> str:
    return html.unescape(raw or "").replace("%%00010", "\n")


class PythonScript:
    @staticmethod
    def parse(node_dir: Optional[Path]) -> PythonScriptSettings:
        if not node_dir:
            return PythonScriptSettings()
        settings_path = node_dir / "settings.xml"
        if not settings_path.exists():
            return PythonScriptSettings()

        root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()
        raw_script = first(root, ".//*[local-name()='config' and @key='model']/*[local-name()='entry' and @key='script']/@value")
        return PythonScriptSettings(script=_decode_script(raw_script or ""))

    @staticmethod
    def emit(settings: PythonScriptSettings) -> List[str]:
        return [
            f"_script = {repr(settings.script)}",
            "df = df.copy()",
            "class _K2PTable:",
            "    def __init__(self, _df):",
            "        self._df = _df",
            "    def to_pandas(self):",
            "        return self._df.copy()",
            "    @staticmethod",
            "    def from_pandas(_df):",
            "        return _K2PTable(_df)",
            "_knio_mod = types.ModuleType('knime.scripting.io')",
            "_knio_mod.input_tables = [_K2PTable(df)]",
            "_knio_mod.output_tables = [None]",
            "_knio_mod.Table = _K2PTable",
            "_knime_mod = types.ModuleType('knime')",
            "_scripting_mod = types.ModuleType('knime.scripting')",
            "_scripting_mod.io = _knio_mod",
            "_knime_mod.scripting = _scripting_mod",
            "_old_modules = {",
            "    'knime': sys.modules.get('knime'),",
            "    'knime.scripting': sys.modules.get('knime.scripting'),",
            "    'knime.scripting.io': sys.modules.get('knime.scripting.io'),",
            "}",
            "sys.modules['knime'] = _knime_mod",
            "sys.modules['knime.scripting'] = _scripting_mod",
            "sys.modules['knime.scripting.io'] = _knio_mod",
            "try:",
            "    exec(_script, {})",
            "finally:",
            "    for _name, _module in _old_modules.items():",
            "        if _module is None:",
            "            sys.modules.pop(_name, None)",
            "        else:",
            "            sys.modules[_name] = _module",
            "_output = _knio_mod.output_tables[0]",
            "out_df = _output.to_pandas() if hasattr(_output, 'to_pandas') else pd.DataFrame(_output)",
        ]


def parse_python_script_settings(node_dir: Optional[Path]) -> PythonScriptSettings:
    return PythonScript.parse(node_dir)


def generate_imports() -> List[str]:
    return ["import pandas as pd", "import sys", "import types"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
    out_ports: Optional[List[str]] = None,
) -> List[str]:
    ndir = Path(node_dir) if node_dir else None
    settings = parse_python_script_settings(ndir)
    src_id, src_port = normalize_in_ports(in_ports)[0] if in_ports else ("UNKNOWN", "1")

    lines: List[str] = [
        f"# {HUB_URL}",
        f"df = context['{src_id}:{src_port}']  # input table",
    ]
    lines.extend(PythonScript.emit(settings))

    for port in sorted({str(p or "1") for p in (out_ports or ["1"])}):
        lines.append(f"context['{node_id}:{port}'] = out_df")
    return lines


def get_name() -> str:
    return "Python Script"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(str(src_id), str(getattr(e, "source_port", "") or "1")) for src_id, e in (incoming or [])]
    out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in (outgoing or [])] or ["1"]
    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
