from __future__ import annotations

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
Robust KNIME component runner for knime2py PEX:
- Reads user format selections from input table.
- Validates flow variables and paths.
- Executes the PEX and returns a single table with `stdout` and `stderr`.
- Converts common failures into actionable guidance for end users.
- If a user-side fix is unlikely, asks the user to report to:
  - https://github.com/users/vitalii-kaplan/projects/1
  - https://forum.knime.com/
"""

import knime.scripting.io as knio

import os
import sys
import platform
import traceback
import subprocess
from pathlib import Path
import pandas as pd


def _as_str(s: object) -> str:
    return "" if s is None else str(s)


def _advise_for_error(stderr: str, context: dict[str, str]) -> str:
    """Append tailored guidance for frequent failure modes."""
    advice: list[str] = []

    # Missing PEX file
    if "PEX not found:" in stderr:
        advice += [
            "Check the 'knime2py PEX' path. It must point to an existing .pex file on the local filesystem.",
            "If you selected a KNIME mountpoint (URI like knime://…), switch the File Chooser to 'Local' and select a real file path.",
        ]

    # Windows privilege / symlink issues
    if "WinError 1314" in stderr or "A required privilege is not held by the client" in stderr:
        advice += [
            "Windows blocked creation of links inside the PEX cache.",
            "Fixes:",
            "  • Use the updated PEX built with --link-mode=copy (recommended).",
            "  • Or enable Windows Developer Mode for your user and retry.",
            "  • As a last resort, run KNIME with elevated privileges.",
        ]

    # Interpreter constraint mismatch
    if "No interpreter compatible with the requested constraints" in stderr or "Version matches CPython" in stderr:
        advice += [
            "Your Python installation does not satisfy the PEX interpreter constraints.",
            "Fixes:",
            "  • Install the required Python version (e.g., Python 3.11) and ensure it is on PATH.",
            "  • Or set the environment variable PEX_PYTHON to the full path of a compatible interpreter.",
            "  • Or ask the developer for a PEX built for your Python version range.",
        ]

    # Permission errors
    if ("Permission denied" in stderr) or ("Access is denied" in stderr):
        advice += [
            "The selected output directory is not writable in your environment.",
            "Choose a different local directory with write permissions.",
        ]

    # KNIME URI misuse
    if "knime://" in context.get("k2p_bin", "") or "knime://" in context.get("input_knime", ""):
        advice += [
            "One or more paths use a KNIME URI (knime://…).",
            "Use 'Local' file system selectors and pick real OS paths for the PEX and workflow directory.",
        ]

    # Workflow structure hint
    if "workflow.knime" not in stderr and context.get("input_knime", ""):
        # If user pointed to a file instead of a workflow folder, hint politely.
        in_path = context["input_knime"]
        try:
            p = Path(in_path)
            if p.exists() and p.is_file() and p.name != "workflow.knime":
                advice += [
                    "The input should be a KNIME workflow directory (containing 'workflow.knime'),",
                    f"but a file was provided: {in_path}",
                    "Select the workflow folder instead.",
                ]
            elif p.exists() and p.is_dir():
                if not (p / "workflow.knime").exists():
                    advice += [
                        f"No 'workflow.knime' found under: {in_path}",
                        "Select a valid KNIME workflow directory.",
                    ]
        except Exception:
            pass

    # Fallback: if nothing matched but we still failed, include generic guidance.
    if advice and "Please report" not in stderr:
        advice += [
            "If the issue persists after trying the steps above, please report it:",
            "  • Project board: https://github.com/users/vitalii-kaplan/projects/1",
            "  • KNIME Forum:  https://forum.knime.com/",
        ]

    if advice:
        return stderr.rstrip() + "\n\n" + "Advice:\n" + "\n".join(advice)
    return stderr


def _safe_fail(message: str, context: dict[str, str]) -> None:
    """Emit a single-row table with empty stdout and informative stderr."""
    err = _advise_for_error(message, context)
    knio.output_tables[0] = knio.Table.from_pandas(pd.DataFrame({"stdout": [""], "stderr": [err]}))


def _normalize_selection_table() -> set[str]:
    df = knio.input_tables[0].to_pandas()
    if df.shape[1] != 1:
        raise ValueError(f"Expected exactly 1 column in the selection table; found {df.shape[1]}.")
    col = df.columns[0]
    # Normalize, drop NaNs/empties, keep only known tokens
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
        raise ValueError(
            "No valid output formats selected. Choose at least one of: .py, .ipynb, .dot, .json."
        )
    return selected


# -------------------- Main guarded execution --------------------
stdout_str: str = ""
stderr_str: str = ""

# Capture minimal context for better error messages
_ctx = {
    "os": platform.platform(),
    "python": sys.version.replace("\n", " "),
    "k2p_bin": _as_str(knio.flow_variables.get("k2p_bin")),
    "input_knime": _as_str(knio.flow_variables.get("k2p_workflow")),
    "output_dir": _as_str(knio.flow_variables.get("output_dir")),
}

try:
    # ---- Inputs ----
    if not _ctx["k2p_bin"]:
        raise ValueError("Flow variable 'k2p_bin' is missing.")
    if not _ctx["input_knime"]:
        raise ValueError("Flow variable 'k2p_workflow' is missing.")
    if not _ctx["output_dir"]:
        raise ValueError("Flow variable 'output_dir' is missing.")

    k2p_bin = _ctx["k2p_bin"]
    input_knime = Path(_ctx["input_knime"]).expanduser()
    output_py = Path(_ctx["output_dir"]).expanduser()

    # Path validations without creating or wiping anything
    pex_path = Path(k2p_bin)
    if not pex_path.is_file():
        _safe_fail(f"PEX not found: {pex_path.resolve()}", _ctx)
        raise SystemExit(0)

    if not input_knime.exists():
        _safe_fail(f"Input path not found: {input_knime}", _ctx)
        raise SystemExit(0)
    if input_knime.is_dir() and not (input_knime / "workflow.knime").exists():
        # Not fatal, but warn in stderr if execution later fails.
        pass

    if not output_py.exists():
        _safe_fail(
            f"Output directory does not exist: {output_py}\n"
            "Create the directory or choose an existing writable location.",
            _ctx,
        )
        raise SystemExit(0)
    if not output_py.is_dir():
        _safe_fail(f"Output path is not a directory: {output_py}", _ctx)
        raise SystemExit(0)

    # ---- Selection table -> args ----
    selected = _normalize_selection_table()
    want_py = ".py" in selected
    want_ipynb = ".ipynb" in selected
    want_dot = ".dot" in selected
    want_json = ".json" in selected

    # Workbooks: both selected => omit flag to generate both
    if want_py and want_ipynb:
        workbook_args: list[str] = []
    elif want_py:
        workbook_args = ["--workbook", "py"]
    elif want_ipynb:
        workbook_args = ["--workbook", "ipynb"]
    else:
        _safe_fail("One of '.py' or '.ipynb' must be selected.", _ctx)
        raise SystemExit(0)

    # Graphs: both selected => omit flag (tool default should produce both)
    if want_dot and want_json:
        graph_args: list[str] = []
    elif want_dot:
        graph_args = ["--graph", "dot"]
    elif want_json:
        graph_args = ["--graph", "json"]
    else:
        graph_args = ["--graph", "off"]

    # ---- Execute PEX ----
    cmd = [
        sys.executable,           # run the PEX via the current interpreter
        str(pex_path),
        str(input_knime),
        "--out",
        str(output_py),
        *workbook_args,
        *graph_args,
    ]
    # Capture output; do not raise on non-zero to allow rich messaging
    proc = subprocess.run(cmd, text=True, capture_output=True)
    stdout_str = proc.stdout or ""
    stderr_str = proc.stderr or ""

    # Enrich stderr with actionable guidance for common issues
    if proc.returncode != 0 or stderr_str:
        stderr_str = _advise_for_error(
            (stderr_str or f"Process returned non-zero exit code: {proc.returncode}").rstrip(),
            _ctx,
        )

except Exception as e:
    # Convert unexpected exceptions into a structured, reportable error
    tb = traceback.format_exc()
    msg = (
        f"Unhandled error in KNIME Python Script: {e}\n"
        f"OS: {_ctx['os']}\n"
        f"Python: {_ctx['python']}\n"
        f"k2p_bin: {_ctx['k2p_bin']}\n"
        f"input_knime: {_ctx['input_knime']}\n"
        f"output_dir: {_ctx['output_dir']}\n"
        f"Traceback:\n{tb}\n"
        "If this looks like a defect, please report it:\n"
        "  • Project board: https://github.com/users/vitalii-kaplan/projects/1\n"
        "  • KNIME Forum:  https://forum.knime.com/\n"
    )
    stderr_str = msg
    stdout_str = ""

# ---- Single output table: columns stdout, stderr ----
knio.output_tables[0] = knio.Table.from_pandas(
    pd.DataFrame({"stdout": [stdout_str], "stderr": [stderr_str]})
)
