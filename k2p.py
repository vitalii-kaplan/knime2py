#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# k2p.py — KNIME → Python/Notebook codegen & graph exporter (CLI entry point)
#
# What this tool does
# -------------------
# • Takes a single KNIME workflow and generates, per isolated subgraph (component):
#     - A “Python workbook” as a Jupyter notebook (.ipynb) and/or a Python script (.py)
#     - Graph JSON (nodes/edges/metadata)
#     - Graphviz .dot (left-to-right)
#
# Input
# -----
# • You can pass either:
#     1) A path to a specific `workflow.knime` file, OR
#     2) A directory that contains exactly one `workflow.knime`
#
# Outputs
# -------
# • Files are written to the directory given by `--out` (default: ./out_graphs).
# • If `--workbook` is omitted, BOTH notebook and script are generated.
# • If `--graph` is omitted, BOTH JSON and DOT are generated.
#
# Usage (examples)
# ----------------
# # Generate everything (graphs + both workbooks)
# python k2p.py /path/to/workflow.knime --out output/
#
# # Point at a project directory that contains exactly one workflow.knime
# python k2p.py /path/to/KNIME_project_dir --out output/
#
# # Only generate a notebook workbook (skip the .py script)
# python k2p.py /path/to/workflow.knime --out output/ --workbook ipynb
#
# # Only generate a Python script workbook (skip the .ipynb)
# python k2p.py /path/to/workflow.knime --out output/ --workbook py
#
# # Control graph artifacts:
# #   --graph dot   → only .dot
# #   --graph json  → only .json
# #   --graph off   → no graph files
# python k2p.py /path/to/workflow.knime --out output/ --graph dot
# python k2p.py /path/to/workflow.knime --out output/ --graph json
# python k2p.py /path/to/workflow.knime --out output/ --graph off
#
# Behavior and notes
# ------------------
# • The tool prints a JSON summary of emitted files to STDOUT on success.
# • The output directory is created if it does not exist.
# • Subgraphs (weakly connected components) of the workflow are emitted separately
#   and suffixed like `__g01`, `__g02`, etc.
#
# Exit codes
# ----------
# 0  success
# 2  bad input path (does not exist / not a workflow / multiple workflows found)
# 3  parsing error while reading the workflow
# 4  empty workflow (no nodes/edges discovered)
#
# Requirements
# ------------
# • Python 3.8+
# • `lxml`, `graphviz` (CLI optional for rendering .dot), and other libs listed in requirements.
#
# -----------------------------------------------------------------------------


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from knime2py.parse_knime import discover_workflows, parse_workflow_components
from knime2py.emitters import (
    write_graph_json,
    write_graph_dot,
    write_workbook_py,
    write_workbook_ipynb,
)

def _resolve_single_workflow(path: Path) -> Path:
    """Return a single workflow.knime path or exit with an error message."""
    p = path.expanduser().resolve()

    if not p.exists():
        print(f"Path does not exist: {p}", file=sys.stderr)
        raise SystemExit(2)

    if p.is_file():
        if p.name != "workflow.knime":
            print(f"Not a workflow.knime file: {p}", file=sys.stderr)
            raise SystemExit(2)
        return p

    # Directory: must contain exactly one workflow.knime
    wfs = discover_workflows(p)
    if not wfs:
        print(f"No workflow.knime found under {p}", file=sys.stderr)
        raise SystemExit(2)
    if len(wfs) > 1:
        sample = "\n".join(f"  - {wf}" for wf in wfs[:10])
        print(
            f"Multiple workflow.knime files found under {p}. "
            f"Pass the exact path to the workflow.knime you want.\nFound:\n{sample}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return wfs[0]


def run_cli(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Parse a single KNIME workflow and emit graph + workbook per isolated subgraph."
    )
    p.add_argument(
        "path",
        type=Path,
        help="Path to a workflow.knime file OR a directory containing exactly one workflow.knime",
    )
    p.add_argument("--out", type=Path, default=Path("out_graphs"), help="Output directory")
    p.add_argument(
        "--workbook",
        choices=["py", "ipynb"],          # None => generate both
        default=None,
        help="Workbook format to generate. Omit to generate both.",
    )
    p.add_argument(
        "--graph",
        choices=["dot", "json", "off"],
        default=None,                     # None => generate both
        help="Which graph file(s) to emit: dot, json, or off. Omit to generate both.",
    )

    args = p.parse_args(argv)

    wf = _resolve_single_workflow(args.path)
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        graphs = parse_workflow_components(wf)  # one WorkflowGraph per isolated component
    except Exception as e:
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
        # If --workbook is omitted, create BOTH. If set, create only the requested one.
        if args.workbook in (None, "py"):
            wb_py = write_workbook_py(g, out_dir)
        if args.workbook in (None, "ipynb"):
            wb_ipynb = write_workbook_ipynb(g, out_dir)

        components.append({
            "workflow_id": g.workflow_id,
            "json": str(j) if j else None,
            "dot": str(d) if d else None,
            "workbook_py": str(wb_py) if wb_py else None,
            "workbook_ipynb": str(wb_ipynb) if wb_ipynb else None,
            "nodes": len(g.nodes),
            "edges": len(g.edges),
        })

    summary = {
        "workflow": str(wf),
        "total_components": len(components),
        "components": components,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
