from __future__ import annotations

import knime.scripting.io as knio

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd

# ---- Inputs ----
k2p_bin = knio.flow_variables["k2p_bin"]
input_knime = Path(knio.flow_variables["k2p_workflow"]).expanduser()
output_py = Path(knio.flow_variables["output_dir"]).expanduser()

# Selection table: 1 column, up to 4 rows with values in {".py",".ipynb",".dot",".json"}
sel_df = knio.input_tables[0].to_pandas()
if sel_df.shape[1] < 1:
    raise ValueError("Expected a single-column selection table with values like .py, .ipynb, .dot, .json.")
col = sel_df.columns[0]
selected = {str(v).strip() for v in sel_df[col].tolist() if pd.notna(v)}

want_py = ".py" in selected
want_ipynb = ".ipynb" in selected
want_dot = ".dot" in selected
want_json = ".json" in selected

# ---- Derive --workbook ----
# If both .py and .ipynb are selected, omit --workbook to generate both.
if want_py and want_ipynb:
    workbook_args = []
elif want_py:
    workbook_args = ["--workbook", "py"]
elif want_ipynb:
    workbook_args = ["--workbook", "ipynb"]
else:
    raise ValueError('One of ".py" or ".ipynb" must be selected')

# ---- Derive --graph ----
# If both .dot and .json are selected, omit --graph entirely (default will produce both).
if want_dot and want_json:
    graph_args = []
elif want_dot:
    graph_args = ["--graph", "dot"]
elif want_json:
    graph_args = ["--graph", "json"]
else:
    graph_args = ["--graph", "off"]

# ---- Output table 1 (index 1): variables -> paths ----
df_paths = pd.DataFrame(
    {
        "variable": ["k2p_bin", "input_knime", "output_py"],
        "path": [k2p_bin, str(input_knime), str(output_py)],
    }
)
knio.output_tables[1] = knio.Table.from_pandas(df_paths)

# ---- Resolve PEX path and run command ----
PEX_BIN = os.environ.get("PEX_BIN", k2p_bin)
pex_path = Path(PEX_BIN)

stdout_str = ""
stderr_str = ""

if not pex_path.is_file():
    stderr_str = f"PEX not found: {pex_path.resolve()}"
else:
    cmd = [
        sys.executable,
        str(pex_path),
        str(input_knime),
        "--out", str(output_py),
        *workbook_args,
        *graph_args,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    stdout_str = proc.stdout or ""
    stderr_str = proc.stderr or ""

# ---- Output table 0 (index 0): one row with both stdout and stderr ----
df_io = pd.DataFrame({"stdout": [stdout_str], "stderr": [stderr_str]})
knio.output_tables[0] = knio.Table.from_pandas(df_io)
