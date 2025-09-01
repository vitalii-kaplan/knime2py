# emitters.py
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Iterator

__all__ = [
    "topo_order",
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

def _derive_title_and_root(nid: str, n) -> tuple[str, str]:
    """
    Return (title, root_id) from folder name like 'CSV Reader (#1)' when available,
    else from node.name, else short class name from node.type, else 'node_<nid>'.
    """
    root_id = nid
    title = None
    if getattr(n, "path", None):
        base = Path(n.path).name  # e.g. "CSV Reader (#1)"
        m = re.match(r"^(.*?)\s*\(#(\d+)\)\s*$", base)
        if m:
            title = m.group(1)
            root_id = m.group(2)
    if not title:
        if getattr(n, "name", None):
            title = n.name
        elif getattr(n, "type", None):
            title = n.type.rsplit(".", 1)[-1]  # short class name
        else:
            title = f"node_{nid}"
    return title, root_id

def _safe_name_from_title(title: str) -> str:
    # Conservative function name slug: letters/digits/underscore only
    return re.sub(r"\W+", "_", title).strip("_") or "node"

def _id_sort_key(s: str) -> Tuple[int, str]:
    return (int(s), "") if s.isdigit() else (10**9, s)

def _build_edge_maps(edges):
    """
    Build incoming/outgoing maps for quick neighbor lookups.
    Returns (incoming_map, outgoing_map) where:
      incoming_map[target_id] -> [Edge...]
      outgoing_map[source_id] -> [Edge...]
    """
    inc = defaultdict(list)
    out = defaultdict(list)
    for e in edges:
        out[e.source].append(e)
        inc[e.target].append(e)
    return inc, out

def _title_for_neighbor(g, nei_id: str) -> str:
    nn = g.nodes.get(nei_id)
    if not nn:
        return nei_id
    t, _ = _derive_title_and_root(nei_id, nn)
    return t

# ----------------------------
# Graph utility used by emitters
# ----------------------------
def topo_order(nodes, edges) -> list[str]:
    indeg = {nid: 0 for nid in nodes}
    adj = defaultdict(list)
    for e in edges:
        if e.source in nodes and e.target in nodes:
            adj[e.source].append(e.target)
            indeg[e.target] += 1
    q = deque([n for n, d in indeg.items() if d == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(nodes):
        remaining = [n for n in nodes if n not in order]
        order.extend(sorted(remaining))
    return order

# ----------------------------
# Unified traversal for emitters
# ----------------------------
def _traverse_nodes(g) -> Iterator[dict]:
    """
    Yield per-node context in topological order:
      {
        "nid": str,
        "node": Node,
        "title": str,
        "root_id": str,
        "state": Optional[str],
        "incoming": list[(src_id, Edge)],
        "outgoing": list[(dst_id, Edge)],
      }
    """
    order = topo_order(g.nodes, g.edges)
    incoming_map, outgoing_map = _build_edge_maps(g.edges)

    for nid in order:
        n = g.nodes[nid]
        title, root_id = _derive_title_and_root(nid, n)
        state = getattr(n, "state", None)

        incoming = [(e.source, e) for e in incoming_map.get(nid, ())]
        outgoing = [(e.target, e) for e in outgoing_map.get(nid, ())]
        incoming.sort(key=lambda x: _id_sort_key(x[0]))
        outgoing.sort(key=lambda x: _id_sort_key(x[0]))

        yield {
            "nid": nid,
            "node": n,
            "title": title,
            "root_id": root_id,
            "state": state,
            "incoming": incoming,
            "outgoing": outgoing,
        }

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

    # Nodes: label = "<title>\n<root_id>", optional fill by state
    for nid, n in g.nodes.items():
        title, root_id = _derive_title_and_root(nid, n)
        label = _esc(f"{title}\n{root_id}")
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

    lines = []
    lines.append("# Auto-generated from KNIME workflow")
    lines.append(f"# workflow: {g.workflow_id}")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("")
    lines.append("# Simple shared context to pass tabular data between sections")
    lines.append("context = {}  # e.g., {'1:output_table': df}")
    lines.append("")

    # Generate functions in unified traversal order
    call_order: List[str] = []
    for ctx in _traverse_nodes(g):
        nid = ctx["nid"]
        n = ctx["node"]
        title = ctx["title"]
        root_id = ctx["root_id"]
        state = ctx["state"] or "UNKNOWN"
        incoming = ctx["incoming"]
        outgoing = ctx["outgoing"]

        safe_name = _safe_name_from_title(title)
        call_order.append(f"node_{nid}_{safe_name}")

        lines.append(f"def node_{nid}_{safe_name}():")
        lines.append(f"    # {title}")
        lines.append(f"    # root: {root_id}")
        lines.append(f"    # state: {state}")

        if incoming:
            lines.append("    # Input port(s):")
            for src_id, e in incoming:
                port = f" [in:{e.target_port}]" if getattr(e, 'target_port', None) else ""
                lines.append(f"    #  - from {src_id} ({_title_for_neighbor(g, src_id)}){port}")

        if outgoing:
            if incoming:
                lines.append("    #")
            lines.append("    # Output port(s):")
            for dst_id, e in outgoing:
                port = f" [out:{e.source_port}]" if getattr(e, 'source_port', None) else ""
                lines.append(f"    #  - to {dst_id} ({_title_for_neighbor(g, dst_id)}){port}")

        lines.append("    # TODO: implement this node translation")
        if n.path:
            lines.append(f"    # original node path: {n.path}")
        lines.append("    # Example: read from context['<src_id>:output'] and write to context['<this_id>:output']")
        lines.append("    pass")
        lines.append("")

    # run_all invoker in the same traversal order
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
    Emit a Jupyter notebook (.ipynb) with one markdown section and one code cell per KNIME node,
    using the unified traversal. Markdown shows a concise title, root id, node state,
    and lists of input/output neighbors by name.
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

    for ctx in _traverse_nodes(g):
        nid = ctx["nid"]
        n = ctx["node"]
        title = ctx["title"]
        root_id = ctx["root_id"]
        state = ctx["state"] or "UNKNOWN"
        incoming = ctx["incoming"]
        outgoing = ctx["outgoing"]

        # Markdown cell
        md_lines = [f"## {title} \\# `{root_id}`", f" State: `{state}`"]
        if incoming:
            if state:
                md_lines.append("")
            md_lines.append(" Input port(s):")
            for src_id, e in incoming:
                port = f" [in:{e.target_port}]" if getattr(e, 'target_port', None) else ""
                md_lines.append(f" - from `{src_id}` ({_title_for_neighbor(g, src_id)}){port}")
        if outgoing:
            if incoming or state:
                md_lines.append("")
            md_lines.append(" Output port(s):")
            for dst_id, e in outgoing:
                port = f" [out:{e.source_port}]" if getattr(e, 'source_port', None) else ""
                md_lines.append(f" - to `{dst_id}` ({_title_for_neighbor(g, dst_id)}){port}")
        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(md_lines) + "\n"})

        # Code cell
        n_type = n.type or ""
        n_path = n.path or ""
        code_src = (
            f"# {title}  [{n_type}]  (node id: {nid})\n"
            f"# state: {state}\n"
            "# TODO: implement this node translation\n"
            + (f"# original node path: {n_path}\n" if n_path else "")
            + "# Example: read from context['<src_id>:output'] and write to context['<this_id>:output']\n"
            "pass\n"
        )
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code_src})

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
