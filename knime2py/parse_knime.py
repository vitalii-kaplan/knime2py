#!/usr/bin/env python3
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
from .xml_utils import XML_PARSER, parse_settings_xml
from .emitters import write_graph_json, write_graph_dot, write_workbook_py, write_workbook_ipynb

@dataclass
class Node:
    id: str
    name: Optional[str] = None
    type: Optional[str] = None
    path: Optional[str] = None

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

def discover_workflows(root: Path) -> List[Path]:
    return sorted((p for p in root.rglob("workflow.knime") if p.is_file()), key=lambda p: str(p))

def _parse_knime5_structure(root, workflow_file: Path) -> Tuple[Dict[str, Node], List[Edge]]:
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    nodes_cont = root.xpath(".//*[local-name()='config' and @key='nodes']")
    conns_cont = root.xpath(".//*[local-name()='config' and @key='connections']")
    nodes_cont = nodes_cont[0] if nodes_cont else None
    conns_cont = conns_cont[0] if conns_cont else None

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
            raw_id = (ncfg.xpath("string(.//*[local-name()='entry' and @key='id']/@value)") or "").strip()
            nid_str = raw_id if raw_id else str(uuid.uuid4())
            if nid_str in nodes:
                nid_str = f"{nid_str}-{uuid.uuid4()}"

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

def _parse_legacy_structure(root: ET._Element, workflow_file: Path) -> Tuple[Dict[str, Node], List[Edge]]:
    raise ValueError(f"Unsupported/legacy workflow format. File: {workflow_file}")

def parse_workflow(workflow_file: Path) -> WorkflowGraph:
    """
    Parse a single workflow.knime into a WorkflowGraph. Supports KNIME 5.x (config tree).
    Legacy (<node>/<connection>) is currently unsupported.
    """
    root = ET.parse(str(workflow_file), parser=XML_PARSER).getroot()
    nodes, edges = _parse_knime5_structure(root, workflow_file)
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