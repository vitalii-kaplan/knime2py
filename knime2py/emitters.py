from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List
from .traverse import depth_order, traverse_nodes, derive_title_and_root
import re
from dataclasses import dataclass
from typing import List, Optional
from knime2py.nodes.registry import get_handlers

from .traverse import (
    traverse_nodes,         
)

from .nodes import *

__all__ = [
    "write_graph_json",
    "write_graph_dot",
    "write_workbook_py",
    "write_workbook_ipynb",
]

@dataclass
class NodeBlock:
    # identity / ordering
    func_name: str
    nid: str
    title: str
    root_id: str

    # state & summaries (already formatted one-liners)
    state: str                      # upper-cased, e.g. "EXECUTED" / "IDLE"
    comment_line: Optional[str]     # e.g. "comments: Remove Ids and time"
    input_line: Optional[str]       # e.g. "Input: [Port 1] 1:1 from CSV Reader #1"
    output_line: Optional[str]      # e.g. "Output: [Port 1] 1350:1 to 4 Missing Value #4"

    # final code body lines (what will go inside the function / code cell)
    code_lines: List[str]


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


def build_workbook_blocks(g) -> tuple[list[NodeBlock], list[str]]:
    """
    Build render-ready blocks for each node (same content used by .py and .ipynb writers).
    Also aggregates per-node imports into a single preamble list for the whole document.

    Returns:
        (blocks, aggregated_imports)
    """
    blocks: List[NodeBlock] = []

    # Collect unique imports across nodes
    aggregated_imports: set[str] = set()

    for ctx in traverse_nodes(g):
        nid = ctx["nid"]
        n = ctx["node"]
        title = ctx["title"]
        root_id = ctx["root_id"]
        state = (ctx["state"] or "UNKNOWN").upper()
        comments = ctx["comments"]
        incoming = ctx["incoming"]     # list[(src_id, Edge)]
        outgoing = ctx["outgoing"]     # list[(dst_id, Edge)]

        safe_name = _safe_name_from_title(title)
        func_name = f"node_{nid}_{safe_name}"

        # ---- one-line comments
        comment_line = None
        if comments:
            oneliner = "; ".join(line for line in str(comments).splitlines() if line.strip())
            if oneliner:
                comment_line = f"comments: {oneliner}"

        # ---- input line (display this node's input port = target_port; context key = src_id:source_port)
        input_line = None
        if incoming:
            ins = []
            for src_id, e in incoming:
                p_in = str(getattr(e, "target_port", "") or "?")    # this node's port number
                p_src = str(getattr(e, "source_port", "") or "?")   # upstream output index (context key)
                src_title = _title_for_neighbor(g, src_id)
                ins.append(f"[Port {p_in}] {src_id}:{p_src} from {src_title} #{src_id}")
            input_line = "Input: " + "; ".join(ins)

        # ---- output line (display this node's output port = source_port; context key = nid:source_port)
        output_line = None
        if outgoing:
            outs = []
            for dst_id, e in outgoing:
                p_out = str(getattr(e, "source_port", "") or "?")
                dst_title = _title_for_neighbor(g, dst_id)
                outs.append(f"[Port {p_out}] {nid}:{p_out} to {dst_id} {dst_title} #{dst_id}")
            output_line = "Output: " + "; ".join(outs)

        # ---- code body lines (plugin-aware)
        code_lines: List[str] = []

        if state == "IDLE":
            # IDLE nodes: only a hub link (if we have a type) + warning
            if n.type:
                hub_url = f"https://hub.knime.com/knime/extensions/org.knime.features.base/latest/{n.type}"
                code_lines.append(f"# {hub_url}")
            else:
                code_lines.append("# Factory class unavailable")
            code_lines.append("# The node is IDLE. Codegen is not possible. Implement this node manually.")
            code_lines.append("pass")
        else:
            res = None
            handlers = get_handlers()  

            if n.type:
                for h in handlers:
                    try:
                        if h.can_handle(n.type):
                            res = h.handle(n.type, nid, n.path, incoming, outgoing)
                            break
                    except Exception as e:
                        # Don’t crash whole build on a single handler; continue to next
                        print(f"[emitters] Handler {getattr(h, '__name__', h)} failed on node {nid}: {e}", file=sys.stderr)
                        continue

            # res is either (imports, body_lines) or None

            if res:
                found_imports, body = res
                aggregated_imports.update(found_imports)
                code_lines.extend(body)

            else:
                # fallback stub
                if n.type:
                    hub_url = f"https://hub.knime.com/knime/extensions/org.knime.features.base/latest/{n.type}"
                    code_lines.append(f"# {hub_url}")
                else:
                    code_lines.append("# Factory class unavailable")
                code_lines.append("# TODO: implement this node")
                code_lines.append("pass")

        blocks.append(NodeBlock(
            func_name=func_name,
            nid=nid,
            title=title,
            root_id=root_id,
            state=state,
            comment_line=comment_line,
            input_line=input_line,
            output_line=output_line,
            code_lines=code_lines,
        ))

    # Return blocks and a sorted list of unique imports
    return blocks, sorted(aggregated_imports)


def write_workbook_py(g, out_dir: Path) -> Path:
    """
    Emit a single, linear Python script (no per-node functions). Each node becomes a
    commented section header followed by its generated code. Results are still stored
    in `context` for debugging/inspection, but code reads from local vars directly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}_workbook.py"

    blocks, imports = build_workbook_blocks(g)

    lines: List[str] = []

    # ---- Unified header (comments) ----
    lines += [
        "# ====================================================================================================",
        "# knime2py — KNIME → Python workbook",
        f"# Workflow: {g.workflow_id}",
        f"# Source:   {g.workflow_path}",
        "#",
        "# This file was generated by the open-source knime2py project (KNIME → Python codegen).",
        "# GitHub: https://github.com/vitaly-chibrikov/knime2py",
        "#",
        "# Notes:",
        "# • The code below is a linear translation of KNIME nodes into Python sections.",
        "# • A lightweight `context` dict is available for debugging/inspection of intermediate tables.",
        "# ====================================================================================================",
        "",
    ]

    # --- aggregated imports at module top
    if imports:
        lines.extend(imports)
        lines.append("")

    # Shared context remains for debugging
    lines.append("# Simple shared context to pass tabular data between sections (for debugging)")
    lines.append("context = {}  # e.g., {'<node_id>:<port>': df}")
    lines.append("")

    # Linear code, one section per node
    for b in blocks:
        lines.append("################################################################################################################################################################")
        lines.append(f"## {b.title} # `{b.root_id}`")
        lines.append(f"# Node state: `{b.state}`")
        if b.input_line:
            lines.append(f"# {b.input_line}")
        if b.output_line:
            lines.append(f"# {b.output_line}")
        if b.comment_line and b.comment_line != b.title:
            lines.append(f"# {b.comment_line}")

        if not b.code_lines:
            lines.append("# TODO: implement this node")
            lines.append("pass")
        else:
            lines.extend(b.code_lines)

        lines.append("")  # blank line between sections

    fp.write_text("\n".join(lines))
    return fp


def write_workbook_ipynb(g, out_dir: Path) -> Path:
    """
    Jupyter notebook: one markdown cell (header + summaries), then an imports cell,
    then a shared context cell, then one code cell per node.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}_workbook.ipynb"

    blocks, imports = build_workbook_blocks(g)

    cells: List[dict] = []

    # ---- Unified header (markdown) ----
    header_md = (
        "# knime2py — KNIME → Python workbook\n\n"
        f"**Workflow:** `{g.workflow_id}`  \n"
        f"**Source:** `{g.workflow_path}`\n\n"
        "This file was generated by the open-source **knime2py** project (KNIME → Python codegen).  \n"
        "**GitHub:** https://github.com/vitaly-chibrikov/knime2py\n\n"
        "**Notes**\n"
        "- The code below is a linear translation of KNIME nodes into Python sections.\n"
        "- A lightweight `context` dict is available for debugging/inspection of intermediate tables.\n"
    )
    cells.append({"cell_type": "markdown", "metadata": {}, "source": header_md})

    # Aggregated imports cell (if any)
    if imports:
        imports_src = "\n".join(imports) + "\n"
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": imports_src
        })

    # Shared context
    context_src = "# Shared context to pass dataframes/tables between nodes (for debugging)\ncontext = {}\n"
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": context_src})

    # Per-node cells
    for b in blocks:
        md_lines = [f"## {b.title} \\# `{b.root_id}`", f" Node state: `{b.state}`  "]
        if b.comment_line:
            md_lines.append(f" {b.comment_line}  ")
        if b.input_line:
            md_lines.append(f" {b.input_line}  ")
        if b.output_line:
            md_lines.append(f" {b.output_line}  ")
        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(md_lines) + "\n"})

        code_src = "\n".join(b.code_lines) + ("\n" if b.code_lines and not b.code_lines[-1].endswith("\n") else "")
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": code_src or "# TODO: implement this node\npass\n"
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
