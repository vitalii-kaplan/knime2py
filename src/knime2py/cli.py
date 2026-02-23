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
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

# NOTE: relative imports because we're now inside the package under src/
from .parse_knime import WorkflowParseError, parse_workflow_components
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


EXIT_CODES = {
    "missing_workflow": 2,
    "missing_settings": 5,
    "invalid_xml": 6,
    "unsupported_workflow": 7,
    "general_failure": 1,
}


class CliError(Exception):
    """Typed CLI error with a stable code and optional details."""

    def __init__(self, code: str, message: str, details: Optional[object] = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def _emit_error(code: str, message: str, details: Optional[object] = None) -> None:
    payload = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    print(json.dumps({"error": payload}, ensure_ascii=False), file=sys.stderr)


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
        raise CliError("missing_workflow", f"Path does not exist: {p}")

    if p.is_file():
        if p.name != "workflow.knime":
            raise CliError("missing_workflow", f"Not a workflow.knime file: {p}")
        return p

    # Directory: only accept a workflow.knime directly inside it (no recursion)
    wf = p / "workflow.knime"
    if not wf.exists() or not wf.is_file():
        raise CliError("missing_workflow", f"No workflow.knime found in directory: {p}")
    return wf


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    max_entries = 5000
    max_file_size = 20 * 1024 * 1024
    max_total_size = 100 * 1024 * 1024
    max_ratio = 100
    total_size = 0

    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > max_entries:
            raise CliError(
                "general_failure",
                "Zip archive has too many entries.",
                details={"entries": len(infos), "max_entries": max_entries},
            )

        for info in infos:
            name = info.filename.replace("\\", "/")
            if not name or name.endswith("/"):
                continue

            if info.file_size > max_file_size:
                raise CliError(
                    "general_failure",
                    "Zip entry exceeds size limit.",
                    details={"member": name, "size": info.file_size, "max_file_size": max_file_size},
                )

            if info.compress_size and info.file_size / max(1, info.compress_size) > max_ratio:
                raise CliError(
                    "general_failure",
                    "Zip entry compression ratio too high.",
                    details={
                        "member": name,
                        "size": info.file_size,
                        "compressed": info.compress_size,
                        "max_ratio": max_ratio,
                    },
                )

            total_size += info.file_size
            if total_size > max_total_size:
                raise CliError(
                    "general_failure",
                    "Zip archive exceeds total size limit.",
                    details={"total_size": total_size, "max_total_size": max_total_size},
                )

            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                raise CliError(
                    "general_failure",
                    "Zip entry is a symlink.",
                    details={"member": name},
                )

            member_path = (dest_dir / name).resolve()
            if dest_dir.resolve() not in member_path.parents:
                raise CliError(
                    "general_failure",
                    "Unsafe path in zip archive.",
                    details={"member": name},
                )
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(member_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


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
        nargs="?",
        type=Path,
        help="Path to a workflow.knime file OR a directory that directly contains workflow.knime",
    )
    parser.add_argument(
        "--in-zip",
        type=Path,
        default=None,
        help="Input bundle.zip containing workflow.knime at the archive root.",
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
    parser.add_argument(
        "--get-handlers",
        action="store_true",
        help="Print the handlers dictionary discovered by knime2py.nodes.registry.get_handlers() and exit.",
    )

    args = parser.parse_args(argv)

    if args.get_handlers:
        from .nodes.registry import get_handlers

        handlers = get_handlers()
        for factory, module in sorted(handlers.items(), key=lambda x: (getattr(x[1], "__name__", ""), x[0])):
            module_name = getattr(module, "__name__", str(module))
            if module_name == "knime2py.nodes.not_implemented":
                continue
            get_name = getattr(module, "get_name", None)
            if callable(get_name):
                display_name = str(get_name())
            else:
                short_name = module_name.rsplit(".", 1)[-1]
                display_name = " ".join(word.capitalize() for word in short_name.split("_"))
            print(f"{display_name},{factory}")
        return 0

    if args.in_zip and args.path:
        _emit_error(
            "general_failure",
            "Provide either a workflow path or --in-zip, not both.",
        )
        return EXIT_CODES["general_failure"]
    if not args.in_zip and not args.path:
        _emit_error(
            "missing_workflow",
            "Missing workflow path (or use --in-zip).",
        )
        return EXIT_CODES["missing_workflow"]

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wf = None
    workflow_id = None
    workflow_path_label = None
    tmp_dir = None

    try:
        if args.in_zip:
            zip_path = args.in_zip.expanduser().resolve()
            if not zip_path.exists():
                raise CliError("missing_workflow", f"Zip file does not exist: {zip_path}")
            tmp_dir = tempfile.TemporaryDirectory()
            tmp_root = Path(tmp_dir.name)
            _safe_extract_zip(zip_path, tmp_root)
            wf = _resolve_single_workflow(tmp_root)
            workflow_id = zip_path.stem
            workflow_path_label = f"{zip_path}::workflow.knime"
        else:
            wf = _resolve_single_workflow(args.path)

        graphs = parse_workflow_components(
            wf,
            strict=True,
            workflow_id=workflow_id,
            workflow_path=workflow_path_label,
        )  # one WorkflowGraph per isolated component
    except CliError as e:
        _emit_error(e.code, str(e), e.details)
        return EXIT_CODES.get(e.code, EXIT_CODES["general_failure"])
    except WorkflowParseError as e:
        _emit_error(e.code, str(e), e.details)
        return EXIT_CODES.get(e.code, EXIT_CODES["general_failure"])
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc(file=sys.stderr)
        _emit_error("general_failure", f"Unhandled error: {e}")
        return EXIT_CODES["general_failure"]
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    if not graphs:
        _emit_error(
            "unsupported_workflow",
            f"No nodes/edges found in workflow: {wf}",
        )
        return EXIT_CODES["unsupported_workflow"]

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
