#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# knime2py.cli — KNIME → Python/Notebook codegen & graph exporter (CLI entry)
# -----------------------------------------------------------------------------

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


def _resolve_single_workflow(path: Path) -> Path:
    """
    Return the path to a single workflow.knime based on the given path.

    Rules:
      - If 'path' is a file, it must be named 'workflow.knime'.
      - If 'path' is a directory, it must contain a file named 'workflow.knime' directly
        (no recursive search).

    Args:
        path (Path): The path to the workflow file or directory.

    Returns:
        Path: The resolved path to the workflow.knime file.

    Raises:
        SystemExit: If the path does not exist or is not a valid workflow file.
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

    Args:
        argv (Optional[list[str]]): The command-line arguments. If None, uses sys.argv.

    Returns:
        int: Exit code indicating success (0) or failure (non-zero).
    """
    p = argparse.ArgumentParser(
        description="Parse a single KNIME workflow and emit graph + workbook per isolated subgraph."
    )
    p.add_argument(
        "path",
        type=Path,
        help="Path to a workflow.knime file OR a directory that directly contains workflow.knime",
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
        if args.workbook in (None, "py"):
            wb_py = write_workbook_py(g, out_dir, blocks, imports)
        if args.workbook in (None, "ipynb"):
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
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> None:
    """
    Console entrypoint used by `pyproject.toml`.

    Args:
        argv (Optional[list[str]]): The command-line arguments. If None, uses sys.argv.
    """
    code = run_cli(argv)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    # Support direct execution: python -m knime2py or python src/knime2py/cli.py
    main(sys.argv[1:])