#!/usr/bin/env python3
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
        choices=["py", "ipynb"],          # removed "both"
        default=None,                     # None => generate both
        help="Workbook format to generate. Omit to generate both.",
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
        j = write_graph_json(g, out_dir)
        d = write_graph_dot(g, out_dir)

        wb_py = wb_ipynb = None
        # If --workbook is omitted, create BOTH. If set, create only the requested one.
        if args.workbook in (None, "py"):
            wb_py = write_workbook_py(g, out_dir)
        if args.workbook in (None, "ipynb"):
            wb_ipynb = write_workbook_ipynb(g, out_dir)

        components.append({
            "workflow_id": g.workflow_id,
            "json": str(j),
            "dot": str(d),
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