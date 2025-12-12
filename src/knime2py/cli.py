#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# knime2py.cli — KNIME → Python/Notebook codegen & graph exporter (CLI entry)
# -----------------------------------------------------------------------------

"""
KNIME workflow CLI parser and exporter.

Overview
----------------------------
This module parses a KNIME workflow and emits graph representations and
workbooks for isolated subgraphs in Python or Jupyter Notebook formats.

Runtime Behavior
----------------------------
Inputs: A single KNIME workflow file (named 'workflow.knime') or a directory
that directly contains 'workflow.knime'.

Outputs: The tool writes output to a target directory, generating JSON and DOT
graph files, as well as Python and Jupyter Notebook workbooks for each isolated
component. It also prints a machine-readable JSON summary to stdout and can
optionally persist that summary to a file.

Key algorithms or mappings: The code handles the parsing of workflow components
and generates corresponding graph representations.

Edge Cases
----------------------------
The code handles cases where no nodes or edges are found in the workflow,
and it raises appropriate errors for invalid paths.

Generated Code Dependencies
----------------------------
The generated notebooks/scripts may require: pandas, numpy, scikit-learn,
imblearn, matplotlib, and lxml. These are dependencies of the emitted code,
not of this CLI itself.

Usage
----------------------------
Typical invocations:
  k2p /path/to/workflow_dir --out out_graphs [--workbook py|ipynb] [--graph dot|json|off]
  k2p --version
  k2p /path/to/workflow_dir --summary-file out_graphs/summary.json
  k2p /path/to/workflow_dir --debug

Configuration
----------------------------
This CLI does not take a direct path to a 'settings.xml'. Per-node settings are
handled within the library layer during parsing.

Limitations
----------------------------
No recursive search for 'workflow.knime' is performed. The input must be a file
named exactly 'workflow.knime' or a directory that directly contains it.

References
----------------------------
Refer to the KNIME documentation for details on workflow structures and node
configurations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# NOTE: relative imports because we're now inside the package under src/
from .parse_knime import parse_workflow_components
from .emitters import (
    write_graph_json,
    write_graph_dot,
    write_workbook_py,
    write_workbook_ipynb,
    build_workbook_blocks,
)


def _infer_version() -> str:
    """Return package version robustly across wheel/PEX/source layouts."""
    # Resolve the installed distribution name from the package path
    dist_name = (__package__ or "knime2py").split(".")[0]
    try:
        try:
            from importlib.metadata import PackageNotFoundError, version  # py3.8+
        except Exception:  # pragma: no cover
            from importlib_metadata import PackageNotFoundError, version  # type: ignore
        v = version(dist_name)
        if v:
            return v
    except Exception:
        pass
    # Fallback to in-package __version__ if present (source checkout)
    try:
        from . import __version__  # type: ignore
        if __version__:
            return str(__version__)
    except Exception:
        pass
    # Last resort
    return "0+unknown"


def _resolve_single_workflow(path: Path) -> Path:
    """
    Return the path to a single workflow.knime based on the given path.

    Rules:
      - If 'path' is a file, it must be named 'workflow.knime'.
      - If 'path' is a directory, it must contain a file named 'workflow.knime' directly
        (no recursive search).
    """
    p = path.expanduser().resolve()

    if not p.exists():
        print(f"Path does not exist: {p}", file=sys.stderr)
        raise SystemExit(2)

    if p.is_file():
        if p.name != "workflow.knime":
            print(f"Not a workflow.knime file: {p}", file=sys.stderr)
            raise SystemExit(2)
        return p

    # Directory: only accept a workflow.knime directly inside it (no recursion)
    wf = p / "workflow.knime"
    if not wf.exists() or not wf.is_file():
        print(f"No workflow.knime found in directory: {p}", file=sys.stderr)
        raise SystemExit(2)
    return wf


def run_cli(argv: Optional[list[str]] = None) -> int:
    """
    Parse command-line arguments and execute the KNIME workflow parsing and exporting.

    Returns an exit code: 0 on success; non-zero on failure.
    """
    parser = argparse.ArgumentParser(
        prog="k2p",
        description="Parse a single KNIME workflow and emit graph + workbook per isolated subgraph.",
    )

    # --version / -V (prints and exits)
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s { _infer_version() }",
        help="Show program version and exit.",
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to a workflow.knime file OR a directory that directly contains workflow.knime",
    )
    parser.add_argument("--out", type=Path, default=Path("out_graphs"), help="Output directory")
    parser.add_argument(
        "--workbook",
        choices=["py", "ipynb"],          # None => generate both
        default=None,
        help="Workbook format to generate. Omit to generate both.",
    )
    parser.add_argument(
        "--graph",
        choices=["dot", "json", "off"],
        default=None,                     # None => generate both
        help="Which graph file(s) to emit: dot, json, or off. Omit to generate both.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Write the JSON summary to this path as well as stdout.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full traceback on errors.",
    )

    args = parser.parse_args(argv)

    wf = _resolve_single_workflow(args.path)
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        graphs = parse_workflow_components(wf)  # one WorkflowGraph per isolated component
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc(file=sys.stderr)
        else:
            print(f"ERROR parsing {wf}: {e}", file=sys.stderr)
        return 3

    if not graphs:
        print(f"No nodes/edges found in workflow: {wf}", file=sys.stderr)
        return 4

    components = []
    for g in graphs:
        # Conditionally emit JSON/DOT based on --graph
        j = d = None
        if args.graph in (None, "json"):
            j = write_graph_json(g, out_dir)
        if args.graph in (None, "dot"):
            d = write_graph_dot(g, out_dir)
        # args.graph == "off" → skip both

        wb_py = wb_ipynb = None

        # Build blocks/imports once
        blocks, imports = build_workbook_blocks(g)

        # --- per-graph summaries
        idle_count = sum(1 for b in blocks if getattr(b, "state", None) == "IDLE")

        # Collect not-implemented node names with factories
        not_impl_names: set[str] = set()
        for b in blocks:
            if getattr(b, "not_implemented", False):
                node = getattr(g, "nodes", {}).get(getattr(b, "nid", None)) if hasattr(g, "nodes") else None
                factory = (
                    getattr(node, "type", None)
                    or getattr(node, "factory", None)
                    or "UNKNOWN"
                )
                title = getattr(b, "title", "UNKNOWN")
                not_impl_names.add(f"{title} ({factory})")

        # Workbooks
        exportable = getattr(g, "exportable", True)

        if exportable and args.workbook in (None, "py"):
            wb_py = write_workbook_py(g, out_dir, blocks, imports)
        if exportable and args.workbook in (None, "ipynb"):
            wb_ipynb = write_workbook_ipynb(g, out_dir, blocks, imports)

        components.append(
            {
                "workflow_id": getattr(g, "workflow_id", None),
                "json": str(j) if j else None,
                "dot": str(d) if d else None,
                "workbook_py": str(wb_py) if wb_py else None,
                "workbook_ipynb": str(wb_ipynb) if wb_ipynb else None,
                "nodes": len(getattr(g, "nodes", {})),
                "edges": len(getattr(g, "edges", [])),
                "idle": idle_count,
                "not_implemented_count": len(not_impl_names),
                "not_implemented_names": sorted(not_impl_names),
            }
        )

    summary = {
        "workflow": str(wf),
        "total_components": len(components),
        "components": components,
    }

    # Always print to stdout
    print(json.dumps(summary, indent=2))

    # Optionally persist to a file
    if args.summary_file:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return 0


def main(argv: Optional[list[str]] = None) -> None:
    """Console entrypoint used by `pyproject.toml`."""
    code = run_cli(argv)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    # Support direct execution: python -m knime2py or python src/knime2py/cli.py
    main(sys.argv[1:])
