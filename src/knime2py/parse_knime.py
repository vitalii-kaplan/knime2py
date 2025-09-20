# knime2py/parse_knime.py
#!/usr/bin/env python3
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

from lxml import etree as ET
from .xml_utils import XML_PARSER, parse_settings_xml


@dataclass
class Node:
    id: str
    name: Optional[str] = None
    type: Optional[str] = None
    path: Optional[str] = None
    # Execution state (maps to KNIME colors): EXECUTED=green, CONFIGURED=yellow, IDLE=red
    state: Optional[Literal["EXECUTED", "CONFIGURED", "IDLE"]] = None
    # Node annotation text (cleaned: %%00010 removed/collapsed)
    comments: Optional[str] = None


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


def _clean_annotation_text(s: str) -> str:
    """
    KNIME encodes line breaks as the literal token '%%00010'.
    We strip/normalize it to a single space and collapse whitespace.
    """
    s = s.replace("%%00010", " ")
    return " ".join(s.split()).strip()


def _read_state_and_annotation_from_settings(settings_ref: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Read <entry key="state" .../> and nodeAnnotation/text from settings.xml.
    Accepts either a settings.xml file path or the node directory path.
    Returns (STATE|None, COMMENTS|None).
    """
    settings = settings_ref / "settings.xml" if settings_ref.is_dir() else settings_ref
    if not settings.exists():
        return None, None

    try:
        root = ET.parse(str(settings), parser=XML_PARSER).getroot()
        # state
        state_vals = root.xpath(".//*[local-name()='entry' and @key='state']/@value")
        state = (state_vals[0].strip().upper() if state_vals and state_vals[0] else None)

        # nodeAnnotation/text
        ann_vals = root.xpath(
            ".//*[local-name()='config' and @key='nodeAnnotation']"
            "/*[local-name()='entry' and @key='text']/@value"
        )
        comments = None
        if ann_vals:
            raw = ann_vals[0] or ""
            comments = _clean_annotation_text(raw)

        return state, (comments or None)
    except Exception:
        return None, None


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
            state: Optional[str] = None
            comments: Optional[str] = None

            if settings_file:
                rel = Path(settings_file)                      # e.g. "Column Filter (#1350)/settings.xml"
                name = rel.parent.name or name                 # folder name
                abs_settings = (workflow_file.parent / rel)    # absolute path to settings.xml
                if abs_settings.exists():
                    node_path = str(abs_settings.parent)
                    nm2, fac2 = parse_settings_xml(abs_settings.parent)
                    name = nm2 or name
                    node_type = fac2 or node_type
                    state, comments = _read_state_and_annotation_from_settings(abs_settings)

            nodes[nid_str] = Node(
                id=nid_str,
                name=name,
                type=node_type,
                path=node_path,
                state=state,
                comments=comments,
            )

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


def _weakly_connected_components(nodes: Dict[str, Node], edges: List[Edge]) -> List[List[str]]:
    if not nodes:
        return []

    adj: Dict[str, set] = {nid: set() for nid in nodes}
    for e in edges:
        if e.source in nodes and e.target in nodes:
            adj[e.source].add(e.target)
            adj[e.target].add(e.source)

    seen: set = set()
    comps: List[List[str]] = []

    for nid in nodes:
        if nid in seen:
            continue
        stack = [nid]
        comp: List[str] = []
        seen.add(nid)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp, key=lambda x: (int(x) if x.isdigit() else float("inf"), x)))
    comps.sort(key=lambda comp: (int(comp[0]) if comp and comp[0].isdigit() else float("inf"), comp[0] if comp else ""))
    return comps


def _split_into_subgraphs(workflow_id: str, workflow_path: str,
                          nodes: Dict[str, Node], edges: List[Edge]) -> List[WorkflowGraph]:
    comps = _weakly_connected_components(nodes, edges)
    if not comps:
        return []

    subgraphs: List[WorkflowGraph] = []
    for idx, comp_nodes in enumerate(comps, start=1):
        node_subset = {nid: nodes[nid] for nid in comp_nodes}
        edge_subset = [e for e in edges if e.source in node_subset and e.target in node_subset]
        sub_id = f"{workflow_id}__g{idx:02d}"
        subgraphs.append(WorkflowGraph(
            workflow_id=sub_id,
            workflow_path=workflow_path,
            nodes=node_subset,
            edges=edge_subset,
        ))
    return subgraphs


def parse_workflow_components(workflow_file: Path) -> List[WorkflowGraph]:
    """
    Parse a single workflow.knime and return one WorkflowGraph per weakly connected component.
    Component IDs are suffixed as '__g01', '__g02', …
    """
    root = ET.parse(str(workflow_file), parser=XML_PARSER).getroot()
    nodes, edges = _parse_knime5_structure(root, workflow_file)
    if not nodes and not edges:
        nodes, edges = _parse_legacy_structure(root, workflow_file)

    base_id = workflow_file.parent.name or workflow_file.parent.as_posix().replace('/', '_')
    return _split_into_subgraphs(base_id, str(workflow_file), nodes, edges)


def parse_workflow(workflow_file: Path) -> WorkflowGraph:
    """
    Backward-compatible parser that returns the combined graph for the workflow.
    Use parse_workflow_components(...) if you need per-component subgraphs.
    """
    root = ET.parse(str(workflow_file), parser=XML_PARSER).getroot()
    nodes, edges = _parse_knime5_structure(root, workflow_file)
    if not nodes and not edges:
        nodes, edges = _parse_legacy_structure(root, workflow_file)

    workflow_id = workflow_file.parent.name or workflow_file.parent.as_posix().replace('/', '_')
    return WorkflowGraph(
        workflow_id=workflow_id,
        workflow_path=str(workflow_file),
        nodes=nodes,
        edges=edges,
    )
