# emitters.py
from __future__ import annotations
import json
import re
from dataclasses import asdict
from pathlib import Path
from collections import defaultdict, deque

__all__ = [
    "topo_order",
    "write_graph_json",
    "write_graph_dot",
    "write_workbook_py",
    "write_workbook_ipynb",
]

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
    import re

    def _derive_title_and_root(nid: str, n) -> tuple[str, str]:
        """Return (title, root_id) from folder name like 'CSV Reader (#1)' when available."""
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

    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}.dot"
    lines = ["digraph knime {", "  rankdir=LR;"]

    # Nodes: label = "<title>\n<root_id>"
    for nid, n in g.nodes.items():
        title, root_id = _derive_title_and_root(nid, n)
        label = _esc(f"{title}\n{root_id}")
        lines.append(f'  "{nid}" [shape=box, style=rounded, label="{label}"];')

    # Edges (unchanged)
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
    order = topo_order(g.nodes, g.edges)

    def _derive_title_and_root(nid: str, n) -> tuple[str, str]:
        """Return (title, root_id) from folder name like 'CSV Reader (#1)' when available."""
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
                title = n.type.rsplit(".", 1)[-1]
            else:
                title = f"node_{nid}"
        return title, root_id

    def _title_for_neighbor(nei_id: str) -> str:
        nn = g.nodes.get(nei_id)
        if not nn:
            return nei_id
        t, _ = _derive_title_and_root(nei_id, nn)
        return t

    def _id_sort_key(s: str):
        return (int(s), "") if s.isdigit() else (10**9, s)

    def _safe_name_from_title(title: str) -> str:
        # conservative: keep letters/digits/underscore
        return re.sub(r"\W+", "_", title).strip("_") or "node"

    lines = []
    lines.append("# Auto-generated from KNIME workflow")
    lines.append(f"# workflow: {g.workflow_id}")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("")
    lines.append("# Simple shared context to pass tabular data between sections")
    lines.append("context = {}  # e.g., {'1:output_table': df}")
    lines.append("")

    for nid in order:
        n = g.nodes[nid]
        title, root_id = _derive_title_and_root(nid, n)
        safe_name = _safe_name_from_title(title)

        # Collect incoming and outgoing edges for this node
        incoming = [(e.source, e) for e in g.edges if e.target == nid]
        outgoing = [(e.target, e) for e in g.edges if e.source == nid]
        incoming.sort(key=lambda x: _id_sort_key(x[0]))
        outgoing.sort(key=lambda x: _id_sort_key(x[0]))

        lines.append(f"def node_{nid}_{safe_name}():")
        # Comments that mirror the notebook markdown
        lines.append(f"    # {title}")
        lines.append(f"    # root: {root_id}")
        if incoming:
            lines.append("    # Input port(s):")
            for src_id, e in incoming:
                port = f" [in:{e.target_port}]" if getattr(e, 'target_port', None) else ""
                lines.append(f"    #  - from {src_id} ({_title_for_neighbor(src_id)}){port}")
        if outgoing:
            if incoming:
                lines.append("    #")  # blank line between sections
            lines.append("    # Output port(s):")
            for dst_id, e in outgoing:
                port = f" [out:{e.source_port}]" if getattr(e, 'source_port', None) else ""
                lines.append(f"    #  - to {dst_id} ({_title_for_neighbor(dst_id)}){port}")

        # Keep the TODO and original path
        lines.append("    # TODO: implement this node translation")
        if n.path:
            lines.append(f"    # original node path: {n.path}")
        lines.append("    # Example: read from context['<src_id>:output'] and write to context['<this_id>:output']")
        lines.append("    pass")
        lines.append("")

    lines.append("def run_all():")
    for nid in order:
        n = g.nodes[nid]
        title, _ = _derive_title_and_root(nid, n)
        safe_name = _safe_name_from_title(title)
        lines.append(f"    node_{nid}_{safe_name}()")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    run_all()")

    fp.write_text("\n".join(lines))
    return fp


def write_workbook_ipynb(g, out_dir: Path) -> Path:
    """
    Emit a Jupyter notebook (.ipynb) with one markdown section and one code cell per KNIME node,
    ordered by the workflow's topological order. The markdown now shows a concise title, root id,
    and lists of input/output neighbors by name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}_workbook.ipynb"
    order = topo_order(g.nodes, g.edges)

    cells = []

    # Title
    title_md = (
        f"# Workflow: {g.workflow_id}\n"
        f"Generated from KNIME workflow at `{g.workflow_path}`\n"
    )
    cells.append({"cell_type": "markdown", "metadata": {}, "source": title_md})

    # Shared context
    context_src = "# Shared context to pass dataframes/tables between nodes\ncontext = {}\n"
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": context_src})

    def _derive_title_and_root(nid: str, n) -> tuple[str, str]:
        """Return (title, root_id) from folder name like 'CSV Reader (#1)' when available."""
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
                title = n.type.rsplit(".", 1)[-1]
            else:
                title = f"node_{nid}"
        return title, root_id

    def _title_for_neighbor(nei_id: str) -> str:
        nn = g.nodes.get(nei_id)
        if not nn:
            return nei_id
        t, _ = _derive_title_and_root(nei_id, nn)
        return t

    def _id_sort_key(s: str):
        return (int(s), "") if s.isdigit() else (10**9, s)

    # One markdown + one code cell per node
    for nid in order:
        n = g.nodes[nid]
        title, root_id = _derive_title_and_root(nid, n)

        # Collect incoming and outgoing edges for this node
        incoming = [(e.source, e) for e in g.edges if e.target == nid]
        outgoing = [(e.target, e) for e in g.edges if e.source == nid]
        incoming.sort(key=lambda x: _id_sort_key(x[0]))
        outgoing.sort(key=lambda x: _id_sort_key(x[0]))

        # Build markdown block
        md_lines = [f"## {title} \# `{root_id}`"]

        if incoming:
            md_lines.append(" Input port(s):")
            for src_id, e in incoming:
                port = f" [in:{e.target_port}]" if getattr(e, 'target_port', None) else ""
                md_lines.append(f" - from `{src_id}` ({_title_for_neighbor(src_id)}){port}")

        if outgoing:
            if incoming:
                md_lines.append("")
            md_lines.append(" Output port(s):")
            for dst_id, e in outgoing:
                port = f" [out:{e.source_port}]" if getattr(e, 'source_port', None) else ""
                md_lines.append(f" - to `{dst_id}` ({_title_for_neighbor(dst_id)}){port}")

        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(md_lines) + "\n"})

        # Code stub (unchanged)
        n_type = n.type or ""
        n_path = n.path or ""
        code_src = (
            f"# {title}  [{n_type}]  (node id: {nid})\n"
            f"# TODO: implement this node translation\n"
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
