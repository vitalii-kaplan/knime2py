from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional
import re

from .traverse import derive_title_and_root, traverse_nodes
from knime2py.nodes.registry import get_handlers

__all__ = [
    "write_graph_json",
    "write_graph_dot",
    "build_workbook_blocks",
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

    not_implemented: bool

    # state & summaries (already formatted one-liners)
    state: str
    comment_line: Optional[str]
    input_line: Optional[str]
    output_line: Optional[str]

    # final code body lines (what will go inside the function / code cell)
    code_lines: List[str]

    # loop metadata
    indent_prefix: str = ""
    loop_role: Optional[str] = None  # "start" | "finish" | None


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
# Graph emitters
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


# ----------------------------
# Workbook block builder (shared by .py and .ipynb writers)
# ----------------------------
def build_workbook_blocks(g) -> tuple[list["NodeBlock"], list[str]]:
    """
    Build render-ready blocks for each node (same content used by .py and .ipynb writers).
    Also aggregates per-node imports into a single preamble list for the whole document.

    Returns:
        (blocks, aggregated_imports)
    """
    aggregated_imports: set[str] = set()
    handlers = get_handlers()  # dict: { FACTORY_ID: module }, may include "" for default

    prepared: List[dict] = []

    # Loop-aware indentation depth for .py generation
    # Increase AFTER a LOOP='start' node; decrease AFTER a LOOP='finish' node
    indent_depth = 0

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

        # ---- input line
        input_line = None
        if incoming:
            ins = []
            for src_id, e in incoming:
                p_in = str(getattr(e, "target_port", "") or "?")
                p_src = str(getattr(e, "source_port", "") or "?")
                src_title = _title_for_neighbor(g, src_id)
                ins.append(f"[Port {p_in}] {src_id}:{p_src} from {src_title} #{src_id}")
            input_line = "Input: " + "; ".join(ins)

        # ---- output line
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
        not_impl_flag = True  # assume not implemented until proven otherwise

        if state == "IDLE":
            # IDLE nodes: only a hub link (if we have a type) + warning
            if getattr(n, "type", None):
                hub_url = f"https://hub.knime.com/knime/extensions/org.knime.features.base/latest/{n.type}"
                code_lines.append(f"# {hub_url}")
            else:
                code_lines.append("# Factory class unavailable")
            code_lines.append("# The node is IDLE. Codegen is not possible. Implement this node manually.")
            code_lines.append("pass")

        # Try factory-specific handler; fall back to default ("")
        mod = None
        default_mod = handlers.get("")
        if getattr(n, "type", None):
            mod = handlers.get(n.type)
            if mod is not None:
                not_impl_flag = False
        if mod is None and default_mod is not None:
            mod = default_mod

        # Loop role detection for indentation (module-level attribute on handler)
        loop_role = getattr(mod, "LOOP", None) if mod is not None else None
        is_loop_start = (loop_role == "start")
        is_loop_finish = (loop_role == "finish")

        if state != "IDLE":
            res = None
            if mod is not None:
                try:
                    res = mod.handle(n.type, nid, n.path, incoming, outgoing)
                except Exception as e:
                    import sys as _sys
                    print(f"[emitters] Handler {getattr(mod, '__name__', mod)} failed on node {nid}: {e}", file=_sys.stderr)
                    res = None

            if res:
                found_imports, body = res
                if found_imports:
                    aggregated_imports.update(found_imports)
                if body:
                    code_lines.extend(body)
            else:
                # fallback stub (no handler or handler failed)
                if getattr(n, "type", None):
                    hub_url = f"https://hub.knime.com/knime/extensions/org.knime.features.base/latest/{n.type}"
                    code_lines.append(f"# {hub_url}")
                else:
                    code_lines.append("# Factory class unavailable")
                code_lines.append("# TODO: implement this node")
                code_lines.append("pass")

        # ---- apply indentation to this node (metadata + body)
        # Start nodes themselves are NOT indented by the new level (but do inherit any outer level),
        # so compute prefix BEFORE updating indent_depth.
        indent_prefix = "    " * indent_depth if indent_depth > 0 else ""

        # Prepend prefix to metadata lines (so writer banner + these lines align)
        if comment_line:
            comment_line = indent_prefix + comment_line
        if input_line:
            input_line = indent_prefix + input_line
        if output_line:
            output_line = indent_prefix + output_line

        # Body lines — prefix each with the same indent
        if code_lines and indent_prefix:
            code_lines = [(indent_prefix + ln) if ln else ln for ln in code_lines]

        prepared.append({
            "func_name": func_name,
            "nid": nid,
            "title": title,
            "root_id": root_id,
            "state": state,
            "comment_line": comment_line,
            "input_line": input_line,
            "output_line": output_line,
            "code_lines": code_lines,
            "not_impl_flag": not_impl_flag,
            "indent_prefix": indent_prefix,  # used by writers for banner/header indentation
            "loop_role": loop_role,          # used by .ipynb writer to group loop cells
        })

        # Update indent depth AFTER placing current node
        if is_loop_start:
            indent_depth += 1
        if is_loop_finish:
            indent_depth = max(0, indent_depth - 1)

    # Create NodeBlock objects
    blocks: List[NodeBlock] = []
    for p in prepared:
        blocks.append(NodeBlock(
            func_name=p["func_name"],
            nid=p["nid"],
            title=p["title"],
            root_id=p["root_id"],
            not_implemented=bool(p["not_impl_flag"]),
            state=p["state"],
            comment_line=p["comment_line"],
            input_line=p["input_line"],
            output_line=p["output_line"],
            code_lines=p["code_lines"],
            indent_prefix=p.get("indent_prefix", ""),
            loop_role=p.get("loop_role"),
        ))

    # Return blocks and a sorted list of unique imports
    return blocks, sorted(aggregated_imports)


# ----------------------------
# Workbook writers
# ----------------------------
def write_workbook_py(
    g,
    out_dir: Path,
    blocks: Optional[List[NodeBlock]] = None,
    imports: Optional[List[str]] = None,
) -> Path:
    """
    Emit a single, linear Python script (no per-node functions). Each node becomes a
    commented section header followed by its generated code. Results are still stored
    in `context` for debugging/inspection, but code reads from local vars directly.

    If `blocks`/`imports` are not provided, they are computed via build_workbook_blocks(g).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}_workbook.py"

    # Compute once if not provided
    if blocks is None or imports is None:
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
        pref = getattr(b, "indent_prefix", "")
        lines.append(pref + "################################################################################################################################################################")
        lines.append(pref + f"## {b.title} # `{b.root_id}`")
        lines.append(pref + f"# Node state: `{b.state}`")
        if b.input_line:
            lines.append(pref + f"# {b.input_line[len(pref):] if b.input_line.startswith(pref) else b.input_line}")
        if b.output_line:
            lines.append(pref + f"# {b.output_line[len(pref):] if b.output_line.startswith(pref) else b.output_line}")
        if b.comment_line and b.comment_line != b.title:
            lines.append(pref + f"# {b.comment_line[len(pref):] if b.comment_line.startswith(pref) else b.comment_line}")

        if not b.code_lines:
            lines.append(pref + "# TODO: implement this node")
            lines.append(pref + "pass")
        else:
            # code_lines already carry the correct prefix
            lines.extend(b.code_lines)

        lines.append("")  # blank line between sections

    fp.write_text("\n".join(lines))
    return fp


def write_workbook_ipynb(
    g,
    out_dir: Path,
    blocks: Optional[List[NodeBlock]] = None,
    imports: Optional[List[str]] = None,
) -> Path:
    """
    Jupyter notebook: one markdown cell per node; code cells are usually one-per-node
    EXCEPT for loop regions: from LOOP='start' to LOOP='finish' (inclusive), all code
    is combined into a single code cell. Nested loops are supported.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}_workbook.ipynb"

    # Compute once if not provided
    if blocks is None or imports is None:
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

    # ---- Loop-aware emission: combine loop code blocks into one cell
    in_loop = False
    loop_depth = 0
    loop_code_accum: List[str] = []

    def _emit_code_cell(code_lines: List[str]):
        code_src = "\n".join(code_lines)
        if code_src and not code_src.endswith("\n"):
            code_src += "\n"
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": code_src or "# TODO: implement this node\npass\n"
        })

    for b in blocks:
        # Markdown per node (always emitted for readability)
        md_lines = [f"## {b.title} \\# `{b.root_id}`", f" Node state: `{b.state}`  "]
        if b.comment_line:
            md_lines.append(f" {b.comment_line}  ")
        if b.input_line:
            md_lines.append(f" {b.input_line}  ")
        if b.output_line:
            md_lines.append(f" {b.output_line}  ")
        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(md_lines) + "\n"})

        # Loop control
        role = getattr(b, "loop_role", None)
        is_start = (role == "start")
        is_finish = (role == "finish")

        if is_start:
            # Opening a (possibly nested) loop: begin/continue accumulation
            if not in_loop:
                in_loop = True
                loop_depth = 1
                loop_code_accum = []
            else:
                loop_depth += 1
            # Always accumulate the start node's code
            loop_code_accum.extend(b.code_lines or [])
            continue

        if in_loop:
            # We are inside a loop: keep accumulating until closing the outermost loop
            loop_code_accum.extend(b.code_lines or [])
            if is_finish:
                loop_depth -= 1
                if loop_depth <= 0:
                    # Close the loop: emit one combined code cell
                    _emit_code_cell(loop_code_accum)
                    in_loop = False
                    loop_depth = 0
                    loop_code_accum = []
            # Do not emit a per-node code cell while inside loop
            continue

        # Outside any loop: normal one-code-cell-per-node behavior
        _emit_code_cell(b.code_lines or [])

    # Safety: if notebook ends while still "in_loop", flush what we have
    if in_loop and loop_code_accum:
        _emit_code_cell(loop_code_accum)

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
