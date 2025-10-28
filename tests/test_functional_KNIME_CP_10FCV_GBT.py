#!/usr/bin/env python3
"""
KNIME to Python code generator and graph exporter.

Overview
----------------------------
This module processes a KNIME workflow and emits corresponding graph and workbook files
for isolated subgraphs within the workflow.

Runtime Behavior
----------------------------
Inputs:
- The module reads a single KNIME workflow file or a directory containing exactly one
  workflow.knime file.

Outputs:
- The generated code writes to context[...] with mappings for input and output ports.
- The output includes JSON and DOT representations of the workflow graph, as well as
  Python and Jupyter Notebook workbooks.

Key algorithms:
- The module utilizes the `discover_workflows` and `parse_workflow_components` functions
  to extract workflow details and generate the corresponding code.

Edge Cases
----------------------------
The code handles various edge cases, including:
- Validating the existence of the workflow file.
- Ensuring that the directory contains exactly one workflow file.
- Reporting errors for invalid paths or empty workflows.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries:
- pandas
- numpy
- Any other libraries used in the generated code, which are not dependencies of this module.

Usage
----------------------------
This module is typically invoked by the knime2py emitter, which processes the workflow
and generates the corresponding outputs. An example of expected context access is:
```python
context['input_table'] = df
```

Node Identity
----------------------------
The module generates code based on the settings defined in the workflow. The KNIME factory
IDs and any special flags are determined from the workflow's configuration.

Configuration
----------------------------
The settings are defined using a `@dataclass`, which includes important fields such as:
- `input_table`: The input DataFrame for processing.
- `output_table`: The resulting DataFrame after processing.

The `parse_*` functions extract these values from the workflow's settings.xml file.

Limitations
----------------------------
Certain options may not be supported or may only approximate KNIME behavior. Users should
refer to the documentation for specific limitations.

References
----------------------------
For more information, refer to the KNIME documentation and the HUB_URL constant if available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# --- Import shim so this file works both as a package module and as a script ---
if __package__ in (None, ""):
    # Running as a script (e.g., python src/knime2py/cli.py) → add .../src to sys.path
    pkg_parent = Path(__file__).resolve().parents[1]  # .../src
    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))
    from knime2py.parse_knime import discover_workflows, parse_workflow_components  # type: ignore
    from knime2py.emitters import (  # type: ignore
        write_graph_json,
        write_graph_dot,
        write_workbook_py,
        write_workbook_ipynb,
        build_workbook_blocks,
    )
else:
    # Normal package-relative imports
    from .parse_knime import discover_workflows, parse_workflow_components
    from .emitters import (
        write_graph_json,
        write_graph_dot,
        write_workbook_py,
        write_workbook_ipynb,
        build_workbook_blocks,
    )


def _resolve_single_workflow(path: Path) -> Path:
    """Return a single workflow.knime path or exit with an error message.

    This function checks if the provided path is a valid workflow.knime file or
    a directory containing exactly one workflow.knime file. If the path is invalid,
    it prints an error message and exits the program.

    Args:
        path (Path): The path to check for a workflow.knime file.

    Returns:
        Path: The resolved path to the workflow.knime file.
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

    # Directory: must contain exactly one workflow.knime
    wfs = discover_workflows(p)
    if not wfs:
        print(f"No workflow.knime found under {p}", file=sys.stderr)
        raise SystemExit(2)
    return wfs[0]


def run_cli(argv: Optional[list[str]] = None) -> int:
    """Parse command-line arguments and run the KNIME workflow processing.

    This function sets up the command-line interface for the script, processes
    the provided workflow.knime file or directory, and generates the corresponding
    graph and workbook files.

    Args:
        argv (Optional[list[str]]): The command-line arguments to parse. If None,
                                     uses sys.argv.

    Returns:
        int: The exit code of the operation (0 for success, non-zero for errors).
    """
    p = argparse.ArgumentParser(
        description="Parse a single KNIME workflow and emit graph + workbook per isolated subgraph."
    )
    # Accept the nonstandard '-help' as an alias (optional, but user-friendly)
    p.add_argument("-help", action="help", help=argparse.SUPPRESS)

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
    """Main entry point for the script.

    This function runs the command-line interface and handles the exit code.

    Args:
        argv (Optional[list[str]]): The command-line arguments to pass to run_cli.
    """
    code = run_cli(argv)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main(sys.argv[1:])
