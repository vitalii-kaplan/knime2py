from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List
from .traverse import depth_order, traverse_nodes, derive_title_and_root
import re

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
        if n.type:
            hub_url = f"https://hub.knime.com/knime/extensions/org.knime.features.base/latest/{n.type}"
            lines.append(f"    # {hub_url}")
        else:
            lines.append("    # Factory class unavailable")

        lines.append(f"    # state: {state}")

        # ---- single-line comments / inputs / outputs ----
        if comments:
            cmnt = " ; ".join(s.strip() for s in str(comments).splitlines() if s.strip())
            if cmnt:
                lines.append(f"    # comments: {cmnt}")

        if incoming:
            ins = []
            for src_id, e in incoming:
                port = f" [in:{e.target_port}]" if getattr(e, 'target_port', None) else ""
                ins.append(f"from {src_id} ({_title_for_neighbor(g, src_id)}){port}")
            lines.append(f"    # Input port(s): " + " ; ".join(ins))

        if outgoing:
            outs = []
            for dst_id, e in outgoing:
                port = f" [out:{e.source_port}]" if getattr(e, 'source_port', None) else ""
                outs.append(f"to {dst_id} ({_title_for_neighbor(g, dst_id)}){port}")
            lines.append(f"    # Output port(s): " + " ; ".join(outs))
        # -----------------------------------------------

        if state == "IDLE":
            lines.append("    # The node is IDLE. Codegen is not possible. Implement this node manually.")
        else:
            lines.append("    # TODO: implement this node")
        lines.append("    pass")
        lines.append("")

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

        # Markdown context with single-line sections
        md_lines: List[str] = [f"## {title} \\# `{root_id}`", f" State: `{state}`"]

        if comments:
            comment_line = " ; ".join([line.strip() for line in str(comments).splitlines() if line.strip()])
            if comment_line:
                md_lines.append(f"\n Comments: {comment_line}")

        if incoming:
            in_parts = []
            for src_id, e in incoming:
                port = f" [in:{e.target_port}]" if getattr(e, 'target_port', None) else ""
                in_parts.append(f"from `{src_id}` ({_title_for_neighbor(g, src_id)}){port}")
            md_lines.append(f"\n Input port(s): " + " ; ".join(in_parts))

        if outgoing:
            out_parts = []
            for dst_id, e in outgoing:
                port = f" [out:{e.source_port}]" if getattr(e, 'source_port', None) else ""
                out_parts.append(f"to `{dst_id}` ({_title_for_neighbor(g, dst_id)}){port}")
            md_lines.append(f"\n Output port(s): " + " ; ".join(out_parts))

        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(md_lines) + "\n"})

        # Short code cell with IDLE handling
        code_lines: List[str] = []
        if n.type:
            hub_url = f"https://hub.knime.com/knime/extensions/org.knime.features.base/latest/{n.type}"
            code_lines.append(f"# {hub_url}")
        else:
            code_lines.append("# Factory class unavailable")
        if state == "IDLE":
            code_lines.append("# The node is IDLE. Codegen is not possible. Implement this node manually.")
        else:
            code_lines.append("# TODO: implement this node")

        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": "\n".join(code_lines)
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
