# emitters.py
from __future__ import annotations
import json
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
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}.dot"
    lines = ["digraph knime {", "  rankdir=LR;"]
    for nid, n in g.nodes.items():
        label_parts = [nid]
        if n.name:
            label_parts.append(n.name)
        if n.type:
            label_parts.append(f"<{n.type}>")
        label = "\n".join(label_parts)
        lines.append(f"  \"{nid}\" [shape=box, style=rounded, label=\"{label}\"];")
    for e in g.edges:
        attrs = []
        if e.source_port:
            attrs.append(f"taillabel=\"{e.source_port}\"")
        if e.target_port:
            attrs.append(f"headlabel=\"{e.target_port}\"")
        attr_str = (" [" + ", ".join(attrs) + "]") if attrs else ""
        lines.append(f"  \"{e.source}\" -> \"{e.target}\"{attr_str};")
    lines.append("}")
    fp.write_text("\n".join(lines))
    return fp

def write_workbook_py(g, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}_workbook.py"
    order = topo_order(g.nodes, g.edges)

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
        safe_name = (n.name or f'node_{nid}').replace(' ', '_').replace('-', '_')
        lines.append(f"def step_{nid}_{safe_name}():")
        lines.append(f"    \"\"\"{n.name or ''}  [{n.type or ''}]  (node id: {nid})\"\"\"")
        lines.append("    # TODO: implement this node translation")
        if n.path:
            lines.append(f"    # original node path: {n.path}")
        lines.append("    # Example: read from context['<src_id>:output'] and write to context['<this_id>:output']")
        lines.append("    pass")
        lines.append("")
    lines.append("def run_all():")
    for nid in order:
        n = g.nodes[nid]
        safe_name = (n.name or f'node_{nid}').replace(' ', '_').replace('-', '_')
        lines.append(f"    step_{nid}_{safe_name}()")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    run_all()")

    fp.write_text("\n".join(lines))
    return fp

def write_workbook_ipynb(g, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}_workbook.ipynb"
    order = topo_order(g.nodes, g.edges)

    cells = []
    title_md = (
        f"# Workflow: {g.workflow_id}\n"
        f"Generated from KNIME workflow at `{g.workflow_path}`\n"
    )
    cells.append({"cell_type": "markdown", "metadata": {}, "source": title_md})
    context_src = "# Shared context to pass dataframes/tables between steps\ncontext = {}\n"
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": context_src})
    for idx, nid in enumerate(order, start=1):
        n = g.nodes[nid]
        n_name = n.name or f"node_{nid}"
        n_type = n.type or ""
        n_path = n.path or ""
        md = (
            f"## Step {idx}: {n_name}\n"
            f"- Node ID: `{nid}`\n"
            f"- Type: `{n_type}`\n"
            f"- Original path: `{n_path}`\n"
        )
        cells.append({"cell_type": "markdown", "metadata": {}, "source": md})
        code_src = (
            f"# {n_name}  [{n_type}]  (node id: {nid})\n"
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
