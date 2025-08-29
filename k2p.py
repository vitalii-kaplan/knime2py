#!/usr/bin/env python3
"""
knime2py: Initial KNIME project parser and graph extractor (updated for KNIME 5.x)

Usage:
  python -m knime2py.parse_knime /path/to/knime_project --out out_dir

What it does (MVP):
  • Recursively discovers KNIME workflows by locating files named 'workflow.knime'.
  • Parses each workflow's XML to extract nodes and connections (edges) for
    KNIME 5.x: nodes/connections are under <config key="nodes"> / <config key="connections">.
  • Augments node metadata from each node's 'settings.xml' when available (e.g., factory/type).
  • Emits a graph JSON and Graphviz .dot per discovered workflow.

Outputs (per workflow):
  • <workflow_id>.json   – nodes, edges, and basic metadata
  • <workflow_id>.dot    – Graphviz representation (left-to-right)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lxml import etree as ET
from xml_utils import XML_PARSER, parse_settings_xml
from emitters import write_graph_json, write_graph_dot, write_workbook_py, write_workbook_ipynb


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
    """Return all paths to 'workflow.knime' under root (recursive), sorted for determinism."""
    return sorted((p for p in root.rglob("workflow.knime") if p.is_file()), key=lambda p: str(p))


def _parse_knime5_structure(root, workflow_file: Path) -> Tuple[Dict[str, Node], List[Edge]]:
    """
    Parse KNIME 5.x style workflow.knime where nodes/connections live under
    <config key="nodes"> / <config key="connections">.
    """
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    # Containers
    nodes_cont = root.xpath(".//*[local-name()='config' and @key='nodes']")
    conns_cont = root.xpath(".//*[local-name()='config' and @key='connections']")
    nodes_cont = nodes_cont[0] if nodes_cont else None
    conns_cont = conns_cont[0] if conns_cont else None

    # Nodes (sorted by integer id when present, else by @key to keep stable order)
    if nodes_cont is not None:
        node_cfgs = nodes_cont.xpath("./*[local-name()='config' and starts-with(@key,'node_')]")

        def node_sort_key(ncfg):
            raw_id = (ncfg.xpath("string(.//*[local-name()='entry' and @key='id']/@value)") or "").strip()
            try:
                n = int(raw_id)
            except Exception:
                n = float("inf")
            key_attr = (ncfg.get("key") or "")
            return (n, key_attr)

        for ncfg in sorted(node_cfgs, key=node_sort_key):
            # id
            raw_id = (ncfg.xpath("string(.//*[local-name()='entry' and @key='id']/@value)") or "").strip()
            nid_str = raw_id if raw_id else str(uuid.uuid4())
            if nid_str in nodes:
                nid_str = f"{nid_str}-{uuid.uuid4()}"  # ensure uniqueness

            # base fields
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

    # Connections (sorted by numeric sourceID, then destID, then ports)
    if conns_cont is not None:
        conn_cfgs = conns_cont.xpath("./*[local-name()='config' and starts-with(@key,'connection_')]")

        def conn_sort_key(ccfg):
            def to_int(s):
                try:
                    return int(s)
                except Exception:
                    return float("inf")
            src = (ccfg.xpath("string(.//*[local-name()='entry' and @key='sourceID']/@value)") or "").strip()
            dst = (ccfg.xpath("string(.//*[local-name()='entry' and @key='destID']/@value)") or "").strip()
            sp = (ccfg.xpath("string(.//*[local-name()='entry' and @key='sourcePort']/@value)") or "").strip()
            dp = (ccfg.xpath("string(.//*[local-name()='entry' and @key='destPort']/@value)") or "").strip()
            return (to_int(src), to_int(dst), sp, dp)

        for ccfg in sorted(conn_cfgs, key=conn_sort_key):
            src = (ccfg.xpath("string(.//*[local-name()='entry' and @key='sourceID']/@value)") or "").strip()
            dst = (ccfg.xpath("string(.//*[local-name()='entry' and @key='destID']/@value)") or "").strip()
            s_port = (ccfg.xpath("string(.//*[local-name()='entry' and @key='sourcePort']/@value)") or "").strip() or None
            d_port = (ccfg.xpath("string(.//*[local-name()='entry' and @key='destPort']/@value)") or "").strip() or None
            if src and dst:
                edges.append(Edge(source=str(src), target=str(dst), source_port=s_port, target_port=d_port))

    return nodes, edges


# ---- Legacy <node>/<connection> parsing (older exports) ----
def _parse_legacy_structure(root: ET._Element, workflow_file: Path) -> Tuple[Dict[str, Node], List[Edge]]:
    # TODO: add legacy support if needed.
    raise ValueError(f"Unsupported/legacy workflow format. File: {workflow_file}")


def parse_workflow(workflow_file: Path) -> WorkflowGraph:
    """
    Parse a single workflow.knime into a WorkflowGraph. Supports KNIME 5.x (config tree).
    Legacy (<node>/<connection>) is currently unsupported.
    """
    root = ET.parse(str(workflow_file), parser=XML_PARSER).getroot()

    nodes, edges = _parse_knime5_structure(root, workflow_file)

    # Report legacy/unsupported if nothing found
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
# CLI
# ----------------------------

def run_cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Parse KNIME project(s) and extract workflow graphs.")
    p.add_argument("root", type=Path, help="Path to KNIME project root directory")
    p.add_argument("--out", type=Path, default=Path("out_graphs"), help="Output directory for JSON/DOT")
    p.add_argument("--toponly", action="store_true", help="Emit only the shallowest (top-level) workflows by path depth")
    p.add_argument("--workbook", choices=["py", "ipynb", "both"], default="ipynb", help="Which workbook format(s) to generate.")

    args = p.parse_args(argv)
    if not args.root.exists():
        p.error(f"Root path does not exist: {args.root}")

    workflows = discover_workflows(args.root)
    if not workflows:
        print("No workflow.knime files found under", args.root, file=sys.stderr)
        return 2

    # Optionally filter to top-level workflows (smallest depth per branch)
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
