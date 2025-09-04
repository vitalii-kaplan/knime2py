from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List
from .traverse import depth_order, traverse_nodes, derive_title_and_root
import re

from .nodes import csv_reader, csv_writer

__all__ = [
    "write_graph_json",
    "write_graph_dot",
    "write_workbook_py",
    "write_workbook_ipynb",
]

# ----------------------------
# Shared helpers
# ----------------------------
def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _title_for_neighbor(g, nei_id: str) -> str:
    nn = g.nodes.get(nei_id)
    if not nn:
        return nei_id
    t, _ = derive_title_and_root(nei_id, nn)
    return t


def _safe_name_from_title(title: str) -> str:
    return re.sub(r"\W+", "_", title).strip("_") or "node"

# ----------------------------
# Emission helpers
# ----------------------------
def write_graph_json(g, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}.json"
    payload = {
        "workflow_id": g.workflow_id,
        "workflow_path": g.workflow_path,
        "nodes": {nid: asdict(n) for nid, n in g.nodes.items()},
        "edges": [asdict(e) for e in g.edges],
    }
    fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return fp


def write_graph_dot(g, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}.dot"
    lines = ["digraph knime {", "  rankdir=LR;"]

    color_map = {
        "EXECUTED": "lightgreen",
        "CONFIGURED": "yellow",
        "IDLE": "red",
    }

    # Nodes: label = "<title>\n<root_id>\n<comments?>", optional fill by state
    for nid, n in g.nodes.items():
        title, root_id = derive_title_and_root(nid, n)
        parts = [title, root_id]
        if getattr(n, "comments", None):
            parts.append(str(n.comments))
        label = _esc("\n".join(parts))
        state = getattr(n, "state", None)
        fill = color_map.get(state)
        if fill:
            lines.append(
                f'  "{nid}" [shape=box, style="rounded,filled", fillcolor="{fill}", label="{label}"];'
            )
        else:
            lines.append(f'  "{nid}" [shape=box, style=rounded, label="{label}"];')

    # Edges
    for e in g.edges:
        attrs = []
        if getattr(e, "source_port", None):
            attrs.append(f'taillabel="{_esc(str(e.source_port))}"')
        if getattr(e, "target_port", None):
            attrs.append(f'headlabel="{_esc(str(e.target_port))}"')
        attr_str = (" [" + ", ".join(attrs) + "]") if attrs else ""
        lines.append(f'  "{e.source}" -> "{e.target}"{attr_str};')

    lines.append("}")
    fp.write_text("\n".join(lines))
    return fp


def write_workbook_py(g, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}_workbook.py"

    lines: List[str] = []
    lines.append("# Auto-generated from KNIME workflow")
    lines.append(f"# workflow: {g.workflow_id}")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("")
    lines.append("# Simple shared context to pass tabular data between sections")
    lines.append("context = {}  # e.g., {'1:output_table': df}")
    lines.append("")

    # Functions are emitted in traversal order and then called by run_all()
    call_order: List[str] = []

    for ctx in traverse_nodes(g):
        nid = ctx["nid"]
        n = ctx["node"]
        title = ctx["title"]
        state = (ctx["state"] or "UNKNOWN").upper()
        comments = ctx["comments"]
        incoming = ctx["incoming"]
        outgoing = ctx["outgoing"]

        safe_name = _safe_name_from_title(title)
        call_order.append(f"node_{nid}_{safe_name}")

        lines.append(f"def node_{nid}_{safe_name}():")
        lines.append(f"    # state: {state}")

        # One-line comments
        if comments:
            oneliner = "; ".join(line for line in str(comments).splitlines() if line.strip())
            if oneliner:
                lines.append(f"    # comments: {oneliner}")

        # One-line input summary
        if incoming:
            ins = []
            for src_id, e in incoming:
                p = str(getattr(e, "target_port", "") or "?")
                src_title = _title_for_neighbor(g, src_id)
                ins.append(f"[Port {p}] {src_id}:{p} from {src_title} #{src_id}")
            lines.append("    # Input: " + "; ".join(ins))

        # One-line output summary
        if outgoing:
            outs = []
            for dst_id, e in outgoing:
                p = str(getattr(e, "source_port", "") or "?")
                dst_title = _title_for_neighbor(g, dst_id)
                outs.append(f"[Port {p}] {nid}:{p} to {dst_title} #{dst_id}")
            lines.append("    # Output: " + "; ".join(outs))

        # Code body
        if state == "IDLE":
            # IDLE nodes: no codegen
            if n.type:
                hub_url = f"https://hub.knime.com/knime/extensions/org.knime.features.base/latest/{n.type}"
                lines.append(f"    # {hub_url}")
            else:
                lines.append("    # Factory class unavailable")
            lines.append("    # The node is IDLE. Codegen is not possible. Implement this node manually.")
            lines.append("    pass")
        else:
            # CSV Reader → generate reader code (publish df to context)
            if n.type and csv_reader.can_handle(n.type):
                out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in outgoing]
                body = csv_reader.generate_py_body(nid, n.path, out_ports)
                for bline in body:
                    lines.append("    " + bline)
            # CSV Writer → generate writer code (consume df from context)
            elif n.type and csv_writer.can_handle(n.type):
                in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in incoming]
                body = csv_writer.generate_py_body(nid, n.path, in_ports)
                for bline in body:
                    lines.append("    " + bline)
            else:
                # Fallback stub
                if n.type:
                    hub_url = f"https://hub.knime.com/knime/extensions/org.knime.features.base/latest/{n.type}"
                    lines.append(f"    # {hub_url}")
                else:
                    lines.append("    # Factory class unavailable")
                lines.append("    # TODO: implement this node")
                lines.append("    pass")

        lines.append("")

    # run_all invoker
    lines.append("def run_all():")
    for fn in call_order:
        lines.append(f"    {fn}()")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    run_all()")

    fp.write_text("\n".join(lines))
    return fp


def write_workbook_ipynb(g, out_dir: Path) -> Path:
    """
    Emit a Jupyter notebook (.ipynb) with one markdown section and one *short* code cell per KNIME node.
    Code cell links to the node's KNIME Hub doc and adds a TODO (or an IDLE warning).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}_workbook.ipynb"

    cells: List[dict] = []
    title_md = (
        f"# Workflow: {g.workflow_id}\n"
        f"Generated from KNIME workflow at `{g.workflow_path}`\n"
    )
    cells.append({"cell_type": "markdown", "metadata": {}, "source": title_md})

    context_src = "# Shared context to pass dataframes/tables between nodes\ncontext = {}\n"
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": context_src})

    for ctx in traverse_nodes(g):
        nid = ctx["nid"]
        n = ctx["node"]
        title = ctx["title"]
        root_id = ctx["root_id"]
        state = (ctx["state"] or "UNKNOWN").upper()
        comments = ctx["comments"]
        incoming = ctx["incoming"]
        outgoing = ctx["outgoing"]

        md_lines = [f"## {title} \\# `{root_id}`", f" State: `{state}`  "]

        # One-line comments with hard break
        if comments:
            oneliner = "; ".join(line for line in str(comments).splitlines() if line.strip())
            if oneliner:
                md_lines.append(f" Comments: {oneliner}  ")

        # One-line input summary with hard break
        if incoming:
            ins = []
            for src_id, e in incoming:
                p = str(getattr(e, "target_port", "") or "?")
                src_title = _title_for_neighbor(g, src_id)
                ins.append(f"[Port {p}] {src_id}:{p} from {src_title} #{src_id}")
            md_lines.append(" Input: " + "; ".join(ins) + "  ")

        # One-line output summary with hard break
        if outgoing:
            outs = []
            for dst_id, e in outgoing:
                p = str(getattr(e, "source_port", "") or "?")
                dst_title = _title_for_neighbor(g, dst_id)
                outs.append(f"[Port {p}] {nid}:{p} to {dst_title} #{dst_id}")
            md_lines.append(" Output: " + "; ".join(outs) + "  ")

        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(md_lines) + "\n"})

        # Code cell
        if state == "IDLE":
            code_lines: List[str] = []
            if n.type:
                hub_url = f"https://hub.knime.com/knime/extensions/org.knime.features.base/latest/{n.type}"
                code_lines.append(f"# {hub_url}")
            else:
                code_lines.append("# Factory class unavailable")
            code_lines.append("# The node is IDLE. Codegen is not possible. Implement this node manually.")
            code_src = "\n".join(code_lines)
        else:
            # CSV Reader
            if n.type and csv_reader.can_handle(n.type):
                out_ports = [str(getattr(e, "source_port", "") or "1") for _, e in outgoing]
                code_src = csv_reader.generate_ipynb_code(nid, n.path, out_ports)
            # CSV Writer
            elif n.type and csv_writer.can_handle(n.type):
                # Use SOURCE ports from incoming edges; those are the context keys upstream wrote.
                in_ports = [(src_id, str(getattr(e, "source_port", "") or "1")) for src_id, e in incoming]
                code_src = csv_writer.generate_ipynb_code(nid, n.path, in_ports)

            else:
                code_lines: List[str] = []
                if n.type:
                    hub_url = f"https://hub.knime.com/knime/extensions/org.knime.features.base/latest/{n.type}"
                    code_lines.append(f"# {hub_url}")
                else:
                    code_lines.append("# Factory class unavailable")
                code_lines.append("# TODO: implement this node")
                code_src = "\n".join(code_lines)

        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": code_src
        })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    fp.write_text(json.dumps(nb, indent=2, ensure_ascii=False))
    return fp
