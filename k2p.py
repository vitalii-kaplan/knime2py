#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from knime2py.parse_knime import parse_workflow, discover_workflows
from knime2py.emitters import (
    write_graph_json,
    write_graph_dot,
    write_workbook_py,
    write_workbook_ipynb,
)

def run_cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Parse KNIME project(s) and extract workflow graphs.")
    p.add_argument("root", type=Path, help="Path to KNIME project root directory")
    p.add_argument("--out", type=Path, default=Path("out_graphs"), help="Output directory for JSON/DOT")
    p.add_argument("--toponly", action="store_true", help="Emit only the shallowest (top-level) workflows by path depth")
    p.add_argument("--workbook", choices=["py", "ipynb", "both"], default="ipynb",
                   help="Which workbook format(s) to generate.")
    args = p.parse_args(argv)

    if not args.root.exists():
        p.error(f"Root path does not exist: {args.root}")

    workflows = discover_workflows(args.root)
    if not workflows:
        print("No workflow.knime files found under", args.root, file=sys.stderr)
        return 2

    # Optionally filter to top-level workflows (smallest depth per branch)
    if args.toponly:
        by_parent: Dict[str, Path] = {}
        for wf in workflows:
            key = str(wf.parent)
            if key not in by_parent or len(str(wf).split(os.sep)) < len(str(by_parent[key]).split(os.sep)):
                by_parent[key] = wf
        workflows = list(by_parent.values())

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for wf in workflows:
        try:
            g = parse_workflow(wf)
        except Exception as e:
            print(f"ERROR parsing {wf}: {e}", file=sys.stderr)
            continue

        j = write_graph_json(g, out_dir)
        d = write_graph_dot(g, out_dir)

        wb_py = wb_ipynb = None
        if args.workbook in ("py", "both"):
            wb_py = write_workbook_py(g, out_dir)
        if args.workbook in ("ipynb", "both"):
            wb_ipynb = write_workbook_ipynb(g, out_dir)

        summaries.append({
            "workflow": str(wf),
            "json": str(j),
            "dot": str(d),
            "workbook_py": str(wb_py) if wb_py else None,
            "workbook_ipynb": str(wb_ipynb) if wb_ipynb else None,
            "nodes": len(g.nodes),
            "edges": len(g.edges),
        })

    print(json.dumps({"workflows": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
