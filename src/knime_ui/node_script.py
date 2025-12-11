"""
KNIME Component runner for the knime2py PEX.

Overview
----------------------------
This script backs the “knime2py launcher” Component. It reads the user’s format
choices from the input table, validates the configured knime2py PEX path and
workflow directory (stored in flow variables), and executes the PEX with the
appropriate `--workbook`/`--graph` settings. The captured stdout/stderr are
returned to KNIME in a single two-column table so the UX node can surface
diagnostics.

Runtime Behavior
----------------------------
Inputs:
- A single-column selection table (first input) listing desired output formats
  such as `.py`, `.ipynb`, `.dot`, `.json`.
- Flow variables: `k2p_bin`, `k2p_workflow`, and `output_dir`.

Outputs:
- A single output table with columns `stdout` and `stderr`, representing the
  PEX process output.

Key logic:
- Validates the selection table and ensures at least one workbook format is
  chosen.
- Builds the knime2py command-line based on the requested formats.
- Executes the PEX and augments known failure messages with actionable
  guidance (e.g., missing interpreter, Windows privilege issues).

Edge Cases
----------------------------
- Raises early errors when the selection table is malformed or lacks `.py` /
  `.ipynb`.
- Adds human-friendly guidance when the PEX binary is missing or execution
  fails due to interpreter/permission problems.

Dependencies
----------------------------
- Relies on KNIME’s scripting bridge (`knime.scripting.io`) and pandas for
  table conversion.

Usage
----------------------------
Configured as part of the KNIME Component: the user supplies the PEX path and
workflow directory via flow variables, and selects output formats inside the
node dialog.

Limitations
----------------------------
- Only exposes the `--workbook` and `--graph` toggles; advanced CLI flags are
  not surfaced.
"""

from __future__ import annotations
"""
Resilient KNIME component runner for knime2py PEX.
- Validates inputs, builds args from the selection table, executes PEX.
- Always returns a single-row table with columns: stdout, stderr.
- No SystemExit; no Path.resolve(); subprocess has a timeout.
"""

import knime.scripting.io as knio

import os
import sys
import platform
import traceback
import subprocess
from pathlib import Path
import pandas as pd

# ---- Tunables ----
PEX_TIMEOUT_SEC = 600  # prevent indefinite hangs if the child process stalls

def _as_str(s: object) -> str:
    return "" if s is None else str(s)

def _advise(stderr: str, ctx: dict[str, str]) -> str:
    adv: list[str] = []

    if "PEX not found:" in stderr:
        adv += [
            "Verify the 'knime2py PEX' path points to an existing .pex on the local filesystem.",
            "If you selected a KNIME URI (knime://...), switch the File Chooser to 'Local' and pick a real OS path.",
        ]

    if "WinError 1314" in stderr or "A required privilege is not held by the client" in stderr:
        adv += [
            "Windows blocked creation of links in the PEX cache.",
            "Fixes:",
            "  • Run KNIME with elevated privileges (in admin mode). Only for the first KNIME2PY component run.",
            "  • Or enable Windows Developer Mode for your user and retry.",
        ]

    if "No interpreter compatible with the requested constraints" in stderr or "Version matches CPython" in stderr:
        adv += [
            "Your Python installation does not satisfy the PEX interpreter constraints.",
            "Fixes:",
            "  • Install a compatible Python (e.g. 3.11) and ensure it is on PATH.",
            "  • Or set PEX_PYTHON to a compatible interpreter path.",
            "  • Or request a PEX built for your Python version range.",
        ]

    if ("Permission denied" in stderr) or ("Access is denied" in stderr):
        adv += [
            "The output directory is not writable. Choose a different local directory with write permission.",
        ]

    if "knime://" in ctx.get("k2p_bin", "") or "knime://" in ctx.get("input_knime", ""):
        adv += [
            "One or more inputs use a KNIME URI (knime://...). Use 'Local' file selectors and real OS paths.",
        ]

    # Workflow folder hint
    in_path = ctx.get("input_knime") or ""
    if in_path:
        try:
            p = Path(in_path)
            if p.exists():
                if p.is_file() and p.name != "workflow.knime":
                    adv += [
                        f"The input should be a KNIME workflow directory (containing 'workflow.knime'), but a file was provided: {in_path}",
                        "Select the workflow folder instead.",
                    ]
                elif p.is_dir() and not (p / "workflow.knime").exists():
                    adv += [
                        f"No 'workflow.knime' found in: {in_path}",
                        "Select a valid KNIME workflow directory.",
                    ]
        except Exception:
            pass

    if adv:
        adv += [
            "If the issue persists after trying the steps above, please report it:",
            "  • Project board: https://github.com/users/vitalii-kaplan/projects/1",
            "  • KNIME Forum:  https://forum.knime.com/",
        ]
        return stderr.rstrip() + "\n\nAdvice:\n" + "\n".join(adv)
    return stderr

def _read_selection() -> set[str]:
    df = knio.input_tables[0].to_pandas()
    if df.shape[1] != 1:
        raise ValueError(f"Expected exactly 1 column in the selection table; found {df.shape[1]}.")
    col = df.columns[0]
    allowed = {".py", ".ipynb", ".dot", ".json"}
    vals = []
    for v in df[col].tolist():
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            vals.append(s)
    selected = set(v for v in vals if v in allowed)
    if not selected:
        raise ValueError("No valid formats selected. Choose at least one of: .py, .ipynb, .dot, .json.")
    return selected

# ---------- main ----------
stdout_str = ""
stderr_str = ""

ctx = {
    "os": platform.platform(),
    "python": sys.version.replace("\n", " "),
    "k2p_bin": _as_str(knio.flow_variables.get("k2p_bin")),
    "input_knime": _as_str(knio.flow_variables.get("k2p_workflow")),
    "output_dir": _as_str(knio.flow_variables.get("output_dir")),
}

try:
    # Flow vars presence
    if not ctx["k2p_bin"]:
        raise ValueError("Flow variable 'k2p_bin' is missing.")
    if not ctx["input_knime"]:
        raise ValueError("Flow variable 'k2p_workflow' is missing.")
    if not ctx["output_dir"]:
        raise ValueError("Flow variable 'output_dir' is missing.")

    # Paths (no resolve())
    pex_path = Path(ctx["k2p_bin"])
    input_knime = Path(ctx["input_knime"]).expanduser()
    output_dir = Path(ctx["output_dir"]).expanduser()

    if not pex_path.is_file():
        stderr_str = _advise(f"PEX not found: {ctx['k2p_bin']}", ctx)
    elif not input_knime.exists():
        stderr_str = _advise(f"Input path not found: {ctx['input_knime']}", ctx)
    elif not output_dir.exists():
        stderr_str = _advise(
            f"Output directory does not exist: {ctx['output_dir']}\nCreate the directory or choose an existing writable location.",
            ctx,
        )
    elif not output_dir.is_dir():
        stderr_str = _advise(f"Output path is not a directory: {ctx['output_dir']}", ctx)
    else:
        # Selection -> args
        selected = _read_selection()
        want_py = ".py" in selected
        want_ipynb = ".ipynb" in selected
        want_dot = ".dot" in selected
        want_json = ".json" in selected

        if want_py and want_ipynb:
            workbook_args: list[str] = []           # omit to get both
        elif want_py:
            workbook_args = ["--workbook", "py"]
        elif want_ipynb:
            workbook_args = ["--workbook", "ipynb"]
        else:
            raise ValueError("One of '.py' or '.ipynb' must be selected.")

        if want_dot and want_json:
            graph_args: list[str] = []              # omit to get both
        elif want_dot:
            graph_args = ["--graph", "dot"]
        elif want_json:
            graph_args = ["--graph", "json"]
        else:
            graph_args = ["--graph", "off"]

        cmd = [
            sys.executable,
            str(pex_path),
            str(input_knime),
            "--out", str(output_dir),
            *workbook_args,
            *graph_args,
        ]

        # Run with timeout to prevent hangs
        try:
            proc = subprocess.run(
                cmd, text=True, capture_output=True, timeout=PEX_TIMEOUT_SEC
            )
            stdout_str = proc.stdout or ""
            stderr_raw = proc.stderr or ""
            if proc.returncode != 0 or stderr_raw:
                stderr_str = _advise(
                    (stderr_raw or f"Process returned non-zero exit code: {proc.returncode}").rstrip(),
                    ctx,
                )
        except subprocess.TimeoutExpired:
            stderr_str = _advise(
                f"PEX execution exceeded {PEX_TIMEOUT_SEC} seconds and was aborted.",
                ctx,
            )

except Exception as e:
    tb = traceback.format_exc()
    stderr_str = (
        f"Unhandled error in KNIME Python Script: {e}\n"
        f"OS: {ctx['os']}\n"
        f"Python: {ctx['python']}\n"
        f"k2p_bin: {ctx['k2p_bin']}\n"
        f"input_knime: {ctx['input_knime']}\n"
        f"output_dir: {ctx['output_dir']}\n"
        f"Traceback:\n{tb}\n"
        "If this looks like a defect, please report it:\n"
        "  • Project board: https://github.com/users/vitalii-kaplan/projects/1\n"
        "  • KNIME Forum:  https://forum.knime.com/\n"
    )
    stdout_str = ""

# Always emit exactly one table
knio.output_tables[0] = knio.Table.from_pandas(
    pd.DataFrame({"stdout": [stdout_str], "stderr": [stderr_str]})
)
