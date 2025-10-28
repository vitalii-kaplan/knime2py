# knime2py/parse_knime.py
#!/usr/bin/env python3
"""
Parse KNIME workflow files and extract their structure.

Overview
----------------------------
This module parses KNIME workflow files to extract nodes and edges, producing a 
graph representation of the workflow. It fits into the knime2py generator pipeline 
by enabling the conversion of KNIME workflows into Python code.

Runtime Behavior
----------------------------
Inputs include DataFrames or context keys that the generated code reads. Outputs 
are written to `context[...]`, with port mappings and types defined by the 
workflow structure. The module implements key algorithms for node processing, 
including handling of various KNIME node types.

Edge Cases
----------------------------
The code implements safeguards for empty or constant columns, NaNs, and class 
imbalances, ensuring robust processing of workflow data.

Generated Code Dependencies
----------------------------
This module requires the following external libraries: lxml. These dependencies 
are required by the generated code, not by this code.

Usage
----------------------------
Typical usage involves invoking this module as part of the workflow parsing 
process. An example of expected context access might be:
```python
data = context['input_table']
```

Node Identity
----------------------------
The module handles various KNIME node types, identified by their unique IDs. 
Special flags include LOOP, which indicates the start or end of a loop in the 
workflow.

Configuration
----------------------------
The `Node` dataclass is used for settings, with important fields including:
- id: Unique identifier for the node.
- name: Optional name of the node.
- type: Optional type of the node.
- path: Optional path to the node's settings.
- state: Execution state of the node (EXECUTED, CONFIGURED, IDLE).
- comments: Optional annotation text for the node.

The `parse_settings_xml` function extracts these values using XPaths from the 
settings.xml file, with fallbacks for missing data.

Limitations
----------------------------
Certain KNIME features may not be fully supported or approximated in the 
conversion process.

References
----------------------------
For more information, refer to the KNIME documentation and search for relevant 
terminology related to workflow parsing and node processing.
"""

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
    """
    Discover all workflow.knime files in the given root directory and return their paths sorted.

    Args:
        root (Path): The root directory to search for workflow files.

    Returns:
        List[Path]: A sorted list of paths to workflow.knime files.
    """
    return sorted((p for p in root.rglob("workflow.knime") if p.is_file()), key=lambda p: str(p))


def _clean_annotation_text(s: str) -> str:
    """
    Clean the annotation text by replacing KNIME encoded line breaks and collapsing whitespace.

    Args:
        s (str): The annotation text to clean.

    Returns:
        str: The cleaned annotation text.
    """
    s = s.replace("%%00010", " ")
    return " ".join(s.split()).strip()


def _read_state_and_annotation_from_settings(settings_ref: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Read the execution state and annotation text from the settings.xml file.

    Args:
        settings_ref (Path): The path to the settings.xml file or its directory.

    Returns:
        tuple[Optional[str], Optional[str]]: A tuple containing the state and comments, or (None, None) if not found.
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
    """
    Parse the structure of a KNIME 5 workflow and extract nodes and edges.

    Args:
        root: The root element of the parsed XML.
        workflow_file (Path): The path to the workflow file.

    Returns:
        Tuple[Dict[str, Node], List[Edge]]: A tuple containing a dictionary of nodes and a list of edges.
    """
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
    """
    Parse the structure of a legacy KNIME workflow format.

    Args:
        root (ET._Element): The root element of the parsed XML.
        workflow_file (Path): The path to the workflow file.

    Raises:
        ValueError: If the workflow format is unsupported.

    Returns:
        Tuple[Dict[str, Node], List[Edge]]: This function is not implemented and will raise an error.
    """
    raise ValueError(f"Unsupported/legacy workflow format. File: {workflow_file}")


def _weakly_connected_components(nodes: Dict[str, Node], edges: List[Edge]) -> List[List[str]]:
    """
    Find weakly connected components in the workflow graph.

    Args:
        nodes (Dict[str, Node]): A dictionary of nodes in the graph.
        edges (List[Edge]): A list of edges in the graph.

    Returns:
        List[List[str]]: A list of weakly connected components, each represented as a list of node IDs.
    """
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
    """
    Split the workflow graph into subgraphs based on weakly connected components.

    Args:
        workflow_id (str): The ID of the workflow.
        workflow_path (str): The path to the workflow file.
        nodes (Dict[str, Node]): A dictionary of nodes in the graph.
        edges (List[Edge]): A list of edges in the graph.

    Returns:
        List[WorkflowGraph]: A list of WorkflowGraph objects representing the subgraphs.
    """
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
    Parse a single workflow.knime file and return one WorkflowGraph per weakly connected component.

    Args:
        workflow_file (Path): The path to the workflow file.

    Returns:
        List[WorkflowGraph]: A list of WorkflowGraph objects, each representing a weakly connected component.
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

    Args:
        workflow_file (Path): The path to the workflow file.

    Returns:
        WorkflowGraph: A WorkflowGraph object representing the entire workflow.
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

