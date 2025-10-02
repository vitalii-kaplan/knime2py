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
    nid: str
    title: str
    root_id: str

    # export coverage flag: True iff NO dedicated exporter exists
    not_implemented: bool

    # state & summaries (already formatted one-liners)
    state: str
    comment_line: Optional[str]
    input_line: Optional[str]
    output_line: Optional[str]

    # final code body lines (what will go inside the code cell)
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


def _banner_lines(b: "NodeBlock") -> List[str]:
    """
    Produce compact, comment-style banners to separate nodes inside combined cells.
    Respects the block's indent_prefix so banners stay inside loops.
    """
    pref = b.indent_prefix or ""
    lines = [
        pref + "################################################################################################################################################################",
        pref + f"## {b.title} # `{b.root_id}`",
        pref + f"# Node state: `{b.state}`",
    ]
    if b.input_line:
        il = b.input_line[len(pref):] if b.input_line.startswith(pref) else b.input_line
        lines.append(pref + f"# {il}")
    if b.output_line:
        ol = b.output_line[len(pref):] if b.output_line.startswith(pref) else b.output_line
        lines.append(pref + f"# {ol}")
    if b.comment_line and b.comment_line != b.title:
        cl = b.comment_line[len(pref):] if b.comment_line.startswith(pref) else b.comment_line
        lines.append(pref + f"# {cl}")
    return lines


def _node_markdown(b: "NodeBlock") -> str:
    """One markdown chunk for a single node."""
    md_lines = [f"## {b.title} \\# `{b.root_id}`", f"Node state: `{b.state}`  "]
    if b.comment_line:
        md_lines.append(f"{b.comment_line}  ")
    if b.input_line:
        md_lines.append(f"{b.input_line}  ")
    if b.output_line:
        md_lines.append(f"{b.output_line}  ")
    return "\n".join(md_lines) + "\n"


def _not_impl_list_for_graph(g, blocks: List[NodeBlock]) -> List[str]:
    """Collect display names for nodes that lack a dedicated exporter (regardless of state)."""
    names: set[str] = set()
    for b in blocks:
        if getattr(b, "not_implemented", False):
            node = getattr(g, "nodes", {}).get(getattr(b, "nid", None))
            factory = (
                getattr(node, "type", None)
                or getattr(node, "factory", None)
                or "UNKNOWN"
            )
            title = getattr(b, "title", "UNKNOWN")
            names.add(f"{title} ({factory})")
    return sorted(names)


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

    # Nodes
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
    """
    aggregated_imports: set[str] = set()
    handlers = get_handlers()  # dict: { FACTORY_ID: module }, may include "" for default

    prepared: List[dict] = []
    indent_depth = 0  # loop-aware indentation depth for .py

    for ctx in traverse_nodes(g):
        nid = ctx["nid"]
        n = ctx["node"]
        title = ctx["title"]
        root_id = ctx["root_id"]
        state = (ctx["state"] or "UNKNOWN").upper()
        comments = ctx["comments"]
        incoming = ctx["incoming"]
        outgoing = ctx["outgoing"]

        # Determine exporter presence (coverage): dedicated handler only
        factory = getattr(n, "type", None)
        specific_mod = handlers.get(factory, None) if factory else None
        default_mod = handlers.get("", None)
        has_dedicated_exporter = specific_mod is not None

        # one-liners
        comment_line = None
        if comments:
            oneliner = "; ".join(line for line in str(comments).splitlines() if line.strip())
            if oneliner:
                comment_line = f"comments: {oneliner}"

        input_line = None
        if incoming:
            ins = []
            for src_id, e in incoming:
                p_in = str(getattr(e, "target_port", "") or "?")
                p_src = str(getattr(e, "source_port", "") or "?")
                src_title = _title_for_neighbor(g, src_id)
                ins.append(f"[Port {p_in}] {src_id}:{p_src} from {src_title} #{src_id}")
            input_line = "Input: " + "; ".join(ins)

        output_line = None
        if outgoing:
            outs = []
            for dst_id, e in outgoing:
                p_out = str(getattr(e, "source_port", "") or "?")
                dst_title = _title_for_neighbor(g, dst_id)
                outs.append(f"[Port {p_out}] {nid}:{p_out} to {dst_id} {dst_title} #{dst_id}")
            output_line = "Output: " + "; ".join(outs)

        code_lines: List[str] = []

        if state == "IDLE":
            # Still show the Hub URL and a stub; node may still be counted as implemented if exporter exists
            if factory:
                hub_url = f"https://hub.knime.com/knime/extensions/org.knime.features.base/latest/{factory}"
                code_lines.append(f"# {hub_url}")
            else:
                code_lines.append("# Factory class unavailable")
            code_lines.append("# The node is IDLE. Codegen is not possible. Implement this node manually or run the node in KNIME.")
            code_lines.append("pass")
        else:
            # Prefer dedicated handler; otherwise fall back to default for codegen convenience
            mod = specific_mod or default_mod
            res = None
            if mod is not None:
                try:
                    res = mod.handle(factory, nid, n.path, incoming, outgoing)
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
                if factory:
                    hub_url = f"https://hub.knime.com/knime/extensions/org.knime.features.base/latest/{factory}"
                    code_lines.append(f"# {hub_url}")
                else:
                    code_lines.append("# Factory class unavailable")
                code_lines.append("# TODO: implement this node")
                code_lines.append("pass")

        # Loop role—use the handler that defines LOOP, preferring the dedicated one
        mod_for_role = specific_mod or default_mod
        loop_role = getattr(mod_for_role, "LOOP", None) if mod_for_role is not None else None
        is_loop_start = (loop_role == "start")
        is_loop_finish = (loop_role == "finish")

        # indentation (for .py)
        indent_prefix = "    " * indent_depth if indent_depth > 0 else ""
        if comment_line:
            comment_line = indent_prefix + comment_line
        if input_line:
            input_line = indent_prefix + input_line
        if output_line:
            output_line = indent_prefix + output_line
        if code_lines and indent_prefix:
            code_lines = [(indent_prefix + ln) if ln else ln for ln in code_lines]

        prepared.append({
            "nid": nid,
            "title": title,
            "root_id": root_id,
            "state": state,
            "comment_line": comment_line,
            "input_line": input_line,
            "output_line": output_line,
            "code_lines": code_lines,
            "not_impl_flag": (not has_dedicated_exporter),
            "indent_prefix": indent_prefix,
            "loop_role": loop_role,
        })

        # update indent AFTER placing current node
        if is_loop_start:
            indent_depth += 1
        if is_loop_finish:
            indent_depth = max(0, indent_depth - 1)

    blocks: List[NodeBlock] = []
    for p in prepared:
        blocks.append(NodeBlock(
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
    Linear .py script: each node -> banner + code (with loop indentation).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}_workbook.py"

    if blocks is None or imports is None:
        blocks, imports = build_workbook_blocks(g)

    # Coverage list for header (based solely on dedicated exporter presence)
    not_impl_names = _not_impl_list_for_graph(g, blocks)

    lines: List[str] = []
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
        "#",
        "# Export coverage:",
    ]
    if not_impl_names:
        lines.append("# • The following nodes have no automatic exporter:")
        for name in not_impl_names:
            lines.append(f"#   - {name}")
    else:
        lines.append("# • All nodes successfully exported.")
    lines += [
        "# ====================================================================================================",
        "",
    ]

    if imports:
        lines.extend(imports)
        lines.append("")

    lines.append("# Simple shared context to pass tabular data between sections (for debugging)")
    lines.append("context = {}  # e.g., {'<node_id>:<port>': df}")
    lines.append("")

    for b in blocks:
        lines.extend(_banner_lines(b))
        if not b.code_lines:
            lines.append((b.indent_prefix or "") + "# TODO: implement this node")
            lines.append((b.indent_prefix or "") + "pass")
        else:
            lines.extend(b.code_lines)
        lines.append("")

    fp.write_text("\n".join(lines))
    return fp


def write_workbook_ipynb(
    g,
    out_dir: Path,
    blocks: Optional[List[NodeBlock]] = None,
    imports: Optional[List[str]] = None,
) -> Path:
    """
    Jupyter notebook:
      • Outside loops: one markdown cell per node + one code cell per node.
      • Inside a loop (from LOOP='start' to LOOP='finish'): a single markdown cell is emitted
        containing the comments for ALL nodes in the loop, followed by ONE code cell that
        contains the combined code for the whole loop region (banners included). Nested loops
        are supported via a stack.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}_workbook.ipynb"

    if blocks is None or imports is None:
        blocks, imports = build_workbook_blocks(g)

    # Coverage list for header (based solely on dedicated exporter presence)
    not_impl_names = _not_impl_list_for_graph(g, blocks)
    if not_impl_names:
        coverage_md = "**Export coverage**\n\n- The following nodes have no automatic exporter:\n" + "\n".join(
            f"- {name}" for name in not_impl_names
        ) + "\n"
    else:
        coverage_md = "**Export coverage**\n\n- All nodes successfully exported.\n"

    cells: List[dict] = []

    # Header cell
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
    cells.append({"cell_type": "markdown", "metadata": {}, "source": header_md if header_md.endswith("\n") else header_md + "\n"})

    # Explicit coverage cell
    cells.append({"cell_type": "markdown", "metadata": {}, "source": coverage_md if coverage_md.endswith("\n") else coverage_md + "\n"})

    # Imports cell
    if imports:
        imports_src = "\n".join(imports) + "\n"
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": imports_src
        })

    # Context cell
    context_src = "# Shared context to pass dataframes/tables between nodes (for debugging)\ncontext = {}\n"
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": context_src})

    # Loop-aware emission using a stack for nested loops
    loop_stack: List[dict] = []  # each: {"md": [str], "code": [str]}

    def _emit_code_cell(code_lines: List[str]):
        src = "\n".join(code_lines)
        if src and not src.endswith("\n"):
            src += "\n"
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src or "# TODO: implement this node\npass\n"
        })

    def _emit_md_cell(md_lines: List[str]):
        md = "\n".join(md_lines)
        cells.append({"cell_type": "markdown", "metadata": {}, "source": md if md.endswith("\n") else md + "\n"})

    for b in blocks:
        role = getattr(b, "loop_role", None)
        is_start = (role == "start")
        is_finish = (role == "finish")

        if is_start:
            # push a new loop context
            loop_stack.append({"md": [], "code": []})
            ctx = loop_stack[-1]
            ctx["md"].append(_node_markdown(b))
            ctx["code"].extend(_banner_lines(b))
            ctx["code"].extend(b.code_lines or [])
            continue

        if loop_stack:
            # inside a loop (possibly nested)
            ctx = loop_stack[-1]
            ctx["md"].append(_node_markdown(b))
            ctx["code"].extend(_banner_lines(b))
            ctx["code"].extend(b.code_lines or [])

            if is_finish:
                finished = loop_stack.pop()
                if loop_stack:
                    # nested: bubble up to parent loop (do not emit yet)
                    parent = loop_stack[-1]
                    parent["md"].extend(finished["md"])
                    parent["code"].extend(finished["code"])
                else:
                    # outermost loop closed: emit one markdown cell + one code cell
                    _emit_md_cell(finished["md"])
                    _emit_code_cell(finished["code"])
            # while in loop, do not emit per-node cells
            continue

        # outside any loop: normal per-node cells
        _emit_md_cell([_node_markdown(b)])
        _emit_code_cell(_banner_lines(b) + (b.code_lines or []))

    # Safety: if we ended still inside a loop, flush the accumulated outermost buffer
    if loop_stack:
        finished = loop_stack[0]
        _emit_md_cell(finished["md"])
        _emit_code_cell(finished["code"])

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
