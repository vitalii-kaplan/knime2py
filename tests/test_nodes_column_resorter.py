from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pandas.testing import assert_frame_equal


repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from knime2py.nodes import column_resorter  # noqa: E402


def _run_resorter(settings_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    incoming = [("DATA_SRC", SimpleNamespace(source_port="1"))]
    outgoing = [("OUT", SimpleNamespace(source_port="1"))]
    imports, body = column_resorter.handle(
        column_resorter.FACTORY,
        "NODE_RESORTER",
        str(settings_dir),
        incoming,
        outgoing,
    )
    code = "\n".join(imports + body)
    env = {
        "context": {"DATA_SRC:1": df.copy()},
        "pd": pd,
    }
    exec(code, env, env)
    return env["context"]["NODE_RESORTER:1"]


def test_column_resorter_parses_inz_fixture_order() -> None:
    settings_dir = repo_root / "tests" / "data" / "INZ_visa_decisions_unification" / "Column Resorter (#1520)"

    settings = column_resorter.parse_column_resorter_settings(settings_dir)

    assert settings.column_order == [
        "Country",
        "Number approved",
        "Number declined",
        column_resorter.UNKNOWN_COLUMN_PLACEHOLDER,
    ]


def test_column_resorter_applies_inz_fixture_order() -> None:
    settings_dir = repo_root / "tests" / "data" / "INZ_visa_decisions_unification" / "Column Resorter (#1520)"
    df = pd.DataFrame(
        {
            "Number declined": [2, 4],
            "Extra": ["x", "y"],
            "Country": ["A", "B"],
            "Number approved": [10, 20],
        }
    )

    result = _run_resorter(settings_dir, df)

    expected = df.loc[:, ["Country", "Number approved", "Number declined", "Extra"]]
    assert_frame_equal(result, expected)


def test_column_resorter_inserts_unknown_columns_at_placeholder(tmp_path: Path) -> None:
    node_dir = tmp_path / "Column Resorter"
    node_dir.mkdir()
    node_dir.joinpath("settings.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig" key="settings.xml">
  <config key="model">
    <config key="ColumnOrder">
      <entry key="array-size" type="xint" value="4"/>
      <entry key="0" type="xstring" value="A"/>
      <entry key="1" type="xstring" value="&lt;any unknown new column&gt;"/>
      <entry key="2" type="xstring" value="B"/>
      <entry key="3" type="xstring" value="C"/>
    </config>
  </config>
</config>
""",
        encoding="utf-8",
    )
    df = pd.DataFrame({"B": [2], "X": [99], "A": [1], "Y": [100], "C": [3]})

    result = _run_resorter(node_dir, df)

    assert list(result.columns) == ["A", "X", "Y", "B", "C"]
