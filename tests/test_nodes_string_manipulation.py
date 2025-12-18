"""Tests for the String Manipulation node handler."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import string_manipulation  # noqa: E402


SETTINGS_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.knime.org/2008/09/XMLConfig http://www.knime.org/XMLConfig_2008_09.xsd"
        key="settings.xml">
    <config key="model">
        <entry key="expression" type="xstring" value="{expression}"/>
        <entry key="replaced_column" type="xstring" value="{replaced_column}"/>
        <entry key="append_column" type="xboolean" value="false"/>
        <entry key="insert_missing_as_null" type="xboolean" value="false"/>
    </config>
</config>
"""


CABIN_VALUES = pd.Series(
    [
        "Deck1/Alpha/Bravo/Charlie",
        "123/456/789/000",
        "X/Y/Z/W",
        pd.NA,
    ],
    dtype="string",
)


def _index_of(text: str, needle: str, occurrence: int | None = None) -> int:
    if occurrence is None or occurrence <= 1:
        occurrence = 1
    if occurrence < 1:
        occurrence = 1
    start = 0
    idx = -1
    for _ in range(occurrence):
        idx = text.find(needle, start)
        if idx == -1:
            break
        start = idx + 1
    return idx


def _substr(text: str, start: int, length: int | None = None) -> str:
    start_idx = int(start) if start is not None else 0
    if length is None:
        return text[start_idx:]
    return text[start_idx : start_idx + int(length)]


def _expected_first(value):
    if pd.isna(value):
        return pd.NA
    text = str(value)
    return text[:1]


def _expected_second(value):
    if pd.isna(value):
        return pd.NA
    text = str(value)
    start = _index_of(text, "/") + 1
    length = _index_of(text, "/", 2) - 2
    return _substr(text, start, length)


def _expected_third(value):
    if pd.isna(value):
        return pd.NA
    text = str(value)
    start = _index_of(text, "/", 3) + 1
    return _substr(text, start, 1)


@pytest.mark.parametrize(
    ("expression", "replaced_column", "expected_fn"),
    [
        ("substr($Cabin$, 0, 1)", "Cabin_1", _expected_first),
        (
            "substr($Cabin$, indexOf($Cabin$, &quot;/&quot;)+1, indexOf($Cabin$,&quot;/&quot; ,2 )-2)",
            "Cabin_2",
            _expected_second,
        ),
        (
            "substr($Cabin$, indexOf($Cabin$,&quot;/&quot; ,3)+1, 1)",
            "Cabin_3",
            _expected_third,
        ),
    ],
)
def test_string_manipulation_handles_examples(tmp_path: Path, expression: str, replaced_column: str, expected_fn):
    """Ensure the handler parses provided XML examples and evaluates expressions row-by-row."""
    node_dir = tmp_path / "StringManip"
    node_dir.mkdir()
    node_dir.joinpath("settings.xml").write_text(
        SETTINGS_TEMPLATE.format(expression=expression, replaced_column=replaced_column),
        encoding="utf-8",
    )

    df = pd.DataFrame({"Cabin": CABIN_VALUES.copy()})
    df[replaced_column] = pd.Series(["legacy"] * len(CABIN_VALUES), dtype="string")

    incoming = [("SRC", SimpleNamespace(source_port="1"))]
    outgoing = [("OUT", SimpleNamespace(source_port="1"))]

    imports, body = string_manipulation.handle(
        "org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory",
        "NODE99",
        str(node_dir),
        incoming,
        outgoing,
    )
    code = "\n".join(imports + body)

    env = {"context": {"SRC:1": df.copy()}, "pd": pd}
    exec(code, env, env)

    result_df = env["context"]["NODE99:1"]
    expected_series = pd.Series([expected_fn(val) for val in CABIN_VALUES], dtype="string")

    assert_series_equal(result_df["Cabin"], df["Cabin"], check_names=False)
    assert_series_equal(result_df[replaced_column].astype("string"), expected_series, check_names=False)


@pytest.mark.parametrize(
    ("expression", "replaced_column", "source_col", "expected"),
    [
        ("regexReplace($Ticket$, &quot;[^a-zA-Z0-9]&quot;,&quot;&quot;)", "ClearedTicket", "Ticket", ["ABC123", "456789"]),
        ("count($Name$, &quot;Mr.&quot;)", "Mr", "Name", [1, 0]),
    ],
)
def test_string_manipulation_regex_and_count(tmp_path: Path, expression: str, replaced_column: str, source_col: str, expected):
    node_dir = tmp_path / "StringManipExtra"
    node_dir.mkdir()
    node_dir.joinpath("settings.xml").write_text(
        SETTINGS_TEMPLATE.format(expression=expression, replaced_column=replaced_column),
        encoding="utf-8",
    )

    df = pd.DataFrame(
        {
            "Ticket": ["ABC-123", "456-789"],
            "Name": ["Mr. Smith", "Jane Doe"],
        }
    )

    incoming = [("SRC", SimpleNamespace(source_port="1"))]
    outgoing = [("OUT", SimpleNamespace(source_port="1"))]

    imports, body = string_manipulation.handle(
        "org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory",
        "NODE_EXTRA",
        str(node_dir),
        incoming,
        outgoing,
    )
    code = "\n".join(imports + body)
    env = {"context": {"SRC:1": df.copy()}, "pd": pd}
    exec(code, env, env)

    result_df = env["context"]["NODE_EXTRA:1"]
    assert result_df[replaced_column].tolist() == expected


def test_string_manipulation_count_literal(tmp_path: Path):
    expression = 'count($Name$, &quot;Mr.&quot;)'
    replaced_column = "MrCount"
    node_dir = tmp_path / "StringManipCount"
    node_dir.mkdir()
    node_dir.joinpath("settings.xml").write_text(
        SETTINGS_TEMPLATE.format(expression=expression, replaced_column=replaced_column),
        encoding="utf-8",
    )

    df = pd.DataFrame(
        {
            "Name": [
                'Romaine, Mr. Charles Hallace ("Mr C Rolmane")',
                "Mr. Smith",
                'Mr Charles "Mr" Doe',
            ]
        }
    )

    incoming = [("SRC", SimpleNamespace(source_port="1"))]
    outgoing = [("OUT", SimpleNamespace(source_port="1"))]

    imports, body = string_manipulation.handle(
        "org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory",
        "NODE_COUNT",
        str(node_dir),
        incoming,
        outgoing,
    )
    code = "\n".join(imports + body)
    env = {"context": {"SRC:1": df.copy()}, "pd": pd}
    exec(code, env, env)

    result_df = env["context"]["NODE_COUNT:1"]
    assert result_df[replaced_column].tolist() == [1, 1, 0]
