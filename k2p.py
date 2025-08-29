#!/usr/bin/env python3
"""
knime2py: Initial KNIME project parser and graph extractor (updated for KNIME 5.x)

Usage:
  python -m knime2py.parse_knime /path/to/knime_project --out out_dir

What it does (MVP):
  • Recursively discovers KNIME workflows by locating files named 'workflow.knime'.
  • Parses each workflow's XML to extract nodes and connections (edges) across KNIME versions.
    - KNIME 5.x: nodes/connections are under <config key="nodes"> / <config key="connections">.
    - Legacy: <node> / <connection> elements.
  • Augments node metadata from each node's 'settings.xml' when available (e.g., factory/type).
  • Emits a graph JSON and Graphviz .dot per discovered workflow.

Outputs (per workflow):
  • <workflow_id>.json   – nodes, edges, and basic metadata
  • <workflow_id>.dot    – Graphviz representation (left‑to‑right)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from lxml import etree as ET
XML_PARSER = ET.XMLParser(remove_comments=True, resolve_entities=False, no_network=True, ns_clean=True, recover=True)


# ----------------------------
# XML helpers
# ----------------------------

def _findall_any(parent, names: Tuple[str, ...]):
    """
    Namespace-agnostic search using XPath local-name().
    Returns a list of elements whose local-name is in `names`.
    """
    expr = " | ".join([f".//*[local-name()='{n}']" for n in names])
    return parent.xpath(expr)

def _get_entry_value_by_key(config_el, key_name: str) -> Optional[str]:
    vals = config_el.xpath(".//*[local-name()='entry' and @key=$k]/@value", k=key_name)
    return vals[0] if vals else None

def parse_settings_xml(node_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns: (name, factory) from settings.xml when present.
    Accepts a node directory or a direct settings.xml path.
    """
    settings = node_dir / "settings.xml"
    if not settings.exists():
        if node_dir.name.endswith(".xml") and node_dir.exists():
            settings = node_dir
        else:
            return (None, None)

    root = ET.parse(str(settings), parser=XML_PARSER).getroot()

    # name candidates
    name_vals = root.xpath(
        ".//*[local-name()='entry' and (@key='name' or @key='label' or @key='node_name')]/@value"
    )
    # factory candidates
    fac_vals = root.xpath(
        ".//*[local-name()='entry' and (@key='factory' or @key='node_factory')]/@value"
    )

    return (name_vals[0] if name_vals else None, fac_vals[0] if fac_vals else None)


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Node:
    id: str
    name: Optional[str] = None
    type: Optional[str] = None  # factory or node class when available
    path: Optional[str] = None  # filesystem path of the node directory (if known)

@dataclass
class Edge:
    source: str
    target: str
    source_port: Optional[str] = None
    target_port: Optional[str] = None

@dataclass
class WorkflowGraph:
    workflow_id: str
    workflow_path: str
    nodes: Dict[str, Node]
    edges: List[Edge]

# ----------------------------
# KNIME discovery & parsing
# ----------------------------

def discover_workflows(root: Path) -> List[Path]:
    """Return all paths to 'workflow.knime' under root (recursive)."""
    return [p for p in root.rglob("workflow.knime") if p.is_file()]

def _parse_knime5_structure(root, workflow_file: Path) -> Tuple[Dict[str, Node], List[Edge]]:
    """Parse KNIME 5.x style workflow.knime where nodes/connections live under <config key="nodes"> / <config key="connections">."""
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    # Containers
    nodes_cont = root.xpath(".//*[local-name()='config' and @key='nodes']")
    conns_cont = root.xpath(".//*[local-name()='config' and @key='connections']")
    nodes_cont = nodes_cont[0] if nodes_cont else None
    conns_cont = conns_cont[0] if conns_cont else None

    # Nodes
    if nodes_cont is not None:
        node_cfgs = nodes_cont.xpath("./*[local-name()='config' and starts-with(@key,'node_')]")
        for ncfg in node_cfgs:
            nid = (ncfg.xpath("string(.//*[local-name()='entry' and @key='id']/@value)") or "").strip()
            nid_str = nid if nid else str(uuid.uuid4())

            settings_file = (ncfg.xpath("string(.//*[local-name()='entry' and @key='node_settings_file']/@value)") or "").strip()
            node_type = (ncfg.xpath("string(.//*[local-name()='entry' and @key='node_type']/@value)") or "").strip() or None

            name = None
            node_path = None
            if settings_file:
                rel = Path(settings_file)
                name = rel.parent.name or name
                abs_settings = (workflow_file.parent / rel)
                if abs_settings.exists():
                    node_path = str(abs_settings.parent)
                    nm2, fac2 = parse_settings_xml(abs_settings.parent)
                    name = nm2 or name
                    node_type = fac2 or node_type

            nodes[nid_str] = Node(id=nid_str, name=name, type=node_type, path=node_path)

    # Connections
    if conns_cont is not None:
        conn_cfgs = conns_cont.xpath("./*[local-name()='config' and starts-with(@key,'connection_')]")
        for ccfg in conn_cfgs:
            src = (ccfg.xpath("string(.//*[local-name()='entry' and @key='sourceID']/@value)") or "").strip()
            dst = (ccfg.xpath("string(.//*[local-name()='entry' and @key='destID']/@value)") or "").strip()
            s_port = (ccfg.xpath("string(.//*[local-name()='entry' and @key='sourcePort']/@value)") or "").strip() or None
            d_port = (ccfg.xpath("string(.//*[local-name()='entry' and @key='destPort']/@value)") or "").strip() or None
            if src and dst:
                edges.append(Edge(source=str(src), target=str(dst), source_port=s_port, target_port=d_port))

    return nodes, edges


# ---- Legacy <node>/<connection> parsing (older exports) ----

def _parse_legacy_structure(root: ET.Element, workflow_file: Path) -> Tuple[Dict[str, Node], List[Edge]]:
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    for node_el in _findall_any(root, ("node",)):
        nid = (
            _get_attr_any(node_el, "id", "nodeID", "nodeId", default=None)
            or str(uuid.uuid4())
        )
        name = _get_attr_any(node_el, "name", "nodeName", "label")
        ntype = _get_attr_any(node_el, "factory", "type", "nodeFactory")

        candidate_dirs = []
        for child in workflow_file.parent.iterdir():
            if child.is_dir() and child.name.strip() == str(nid).strip():
                candidate_dirs.append(child)
        if not candidate_dirs:
            for child in workflow_file.parent.iterdir():
                if child.is_dir() and str(nid) in child.name:
                    candidate_dirs.append(child)

        for ndir in candidate_dirs:
            nm2, fac2 = parse_settings_xml(ndir)
            name = name or nm2
            ntype = ntype or fac2
            if name or ntype:
                node_path = str(ndir)
                break
        else:
            node_path = None

        nodes[str(nid)] = Node(id=str(nid), name=name, type=ntype, path=node_path)

    for conn_el in _findall_any(root, ("connection",)):
        src = _get_attr_any(conn_el, "sourceID", "srcId", "source", "from")
        dst = _get_attr_any(conn_el, "destID", "dstId", "target", "to")
        s_port = _get_attr_any(conn_el, "sourcePort", "srcPort", "outport", "outPort")
        d_port = _get_attr_any(conn_el, "destPort", "dstPort", "inport", "inPort")
        if src and dst:
            edges.append(Edge(source=str(src), target=str(dst), source_port=s_port, target_port=d_port))

    return nodes, edges


def parse_workflow(workflow_file: Path) -> WorkflowGraph:
    """Parse a single workflow.knime into a WorkflowGraph. Supports KNIME 5.x (config tree)
    and older structures with <node>/<connection> elements."""
    root = ET.parse(str(workflow_file), parser=XML_PARSER).getroot()

    nodes, edges = _parse_knime5_structure(root, workflow_file)

    # Fallback to legacy if nothing found
    if not nodes and not edges:
        nodes, edges = _parse_legacy_structure(root, workflow_file)

    rel = workflow_file.parent
    workflow_id = rel.name or rel.as_posix().replace('/', '_')

    return WorkflowGraph(
        workflow_id=workflow_id,
        workflow_path=str(workflow_file),
        nodes=nodes,
        edges=edges,
    )

# ----------------------------
# Graph utilities
# ----------------------------
from collections import defaultdict, deque

def topo_order(nodes: Dict[str, Node], edges: List[Edge]) -> List[str]:
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
    # If cycle: fall back to arbitrary stable order of remaining
    if len(order) != len(nodes):
        remaining = [n for n in nodes if n not in order]
        order.extend(sorted(remaining))
    return order


# ----------------------------
# Emission helpers
# ----------------------------

def write_graph_json(g: WorkflowGraph, out_dir: Path) -> Path:
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


def write_graph_dot(g: WorkflowGraph, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}.dot"
    lines = ["digraph knime {", "  rankdir=LR;"]
    # Nodes
    for nid, n in g.nodes.items():
        label_parts = [nid]
        if n.name:
            label_parts.append(n.name)
        if n.type:
            label_parts.append(f"<{n.type}>")
        label = "\n".join(label_parts)
        lines.append(f"  \"{nid}\" [shape=box, style=rounded, label=\"{label}\"];")
    # Edges
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

def write_workbook_py(g: WorkflowGraph, out_dir: Path) -> Path:
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

def write_workbook_ipynb(g: WorkflowGraph, out_dir: Path) -> Path:
    """
    Emit a Jupyter notebook (.ipynb) with one markdown section and one code cell per KNIME node,
    ordered by the workflow's topological order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{g.workflow_id}_workbook.ipynb"
    order = topo_order(g.nodes, g.edges)

    cells = []

    # Top-level title / metadata
    title_md = (
        f"# Workflow: {g.workflow_id}\n"
        f"Generated from KNIME workflow at `{g.workflow_path}`\n"
    )
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": title_md
    })

    # Shared context cell
    context_src = (
        "# Shared context to pass dataframes/tables between steps\n"
        "context = {}\n"
    )
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": context_src
    })

    # One markdown + one code cell per node
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
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": md
        })

        code_src = (
            f"# {n_name}  [{n_type}]  (node id: {nid})\n"
            f"# TODO: implement this node translation\n"
            + (f"# original node path: {n_path}\n" if n_path else "")
            + "# Example: read from context['<src_id>:output'] and write to context['<this_id>:output']\n"
            "pass\n"
        )
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
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    fp.write_text(json.dumps(nb, indent=2, ensure_ascii=False))
    return fp
  

# ----------------------------
# CLI
# ----------------------------

def run_cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Parse KNIME project(s) and extract workflow graphs.")
    p.add_argument("root", type=Path, help="Path to KNIME project root directory")
    p.add_argument("--out", type=Path, default=Path("out_graphs"), help="Output directory for JSON/DOT")
    p.add_argument("--toponly", action="store_true", help="Emit only the shallowest (top‑level) workflows by path depth")
    p.add_argument("--workbook", choices=["py", "ipynb", "both"], default="ipynb", help="Which workbook format(s) to generate.")

  
    args = p.parse_args(argv)
    if not args.root.exists():
        p.error(f"Root path does not exist: {args.root}")

    workflows = discover_workflows(args.root)
    if not workflows:
        print("No workflow.knime files found under", args.root, file=sys.stderr)
        return 2

    # Optionally filter to top‑level workflows (smallest depth per branch)
    if args.toponly:
        by_parent: Dict[str, Path] = {}
        for wf in workflows:
            key = str(wf.parent)
            if key not in by_parent or len(str(wf).split(os.sep)) < len(str(by_parent[key]).split(os.sep)):
                by_parent[key] = wf
        workflows = list(by_parent.values())

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for wf in workflows:
        try:
            g = parse_workflow(wf)
        except Exception as e:
            print(f"ERROR parsing {wf}: {e}", file=sys.stderr)
            continue
        j = write_graph_json(g, out_dir)
        d = write_graph_dot(g, out_dir)

        wb_py = wb_ipynb = None
        if args.workbook in ("py", "both"):
            wb_py = write_workbook_py(g, out_dir)
        if args.workbook in ("ipynb", "both"):
            wb_ipynb = write_workbook_ipynb(g, out_dir)

        summaries.append({
            "workflow": str(wf),
            "json": str(j),
            "dot": str(d),
            "workbook_py": str(wb_py) if wb_py else None,
            "workbook_ipynb": str(wb_ipynb) if wb_ipynb else None,
            "nodes": len(g.nodes),
            "edges": len(g.edges),
        })


    print(json.dumps({"workflows": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
