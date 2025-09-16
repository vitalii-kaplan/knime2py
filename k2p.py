#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# k2p.py — KNIME → Python/Notebook codegen & graph exporter (CLI entry point)
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
    build_workbook_blocks,
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

        # Build blocks/imports once
        blocks, imports = build_workbook_blocks(g)

        # --- per-graph summaries
        idle_count = sum(1 for b in blocks if b.state == "IDLE")

        # List of not-implemented node *names with factories*, e.g.
        # "Row Filter: org.knime.base.node.preproc.filter.row3.RowFilterNodeFactory"
        not_impl_names: set[str] = set()
        for b in blocks:
            if getattr(b, "not_implemented", False):
                node = g.nodes.get(b.nid) if hasattr(g, "nodes") else None
                factory = getattr(node, "type", None) or getattr(node, "factory", None) or "UNKNOWN"
                not_impl_names.add(f"{b.title} ({factory})")

        # Workbooks
        if args.workbook in (None, "py"):
            wb_py = write_workbook_py(g, out_dir, blocks, imports)
        if args.workbook in (None, "ipynb"):
            wb_ipynb = write_workbook_ipynb(g, out_dir, blocks, imports)

        components.append({
            "workflow_id": g.workflow_id,
            "json": str(j) if j else None,
            "dot": str(d) if d else None,
            "workbook_py": str(wb_py) if wb_py else None,
            "workbook_ipynb": str(wb_ipynb) if wb_ipynb else None,
            "nodes": len(g.nodes),
            "edges": len(g.edges),
            "idle": idle_count,
            "not_implemented_count": len(not_impl_names),
            "not_implemented_names": sorted(not_impl_names),
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
