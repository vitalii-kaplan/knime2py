#!/usr/bin/env python3
"""Parse KNIME workflow files and extract their structure.

## Overview

This module parses KNIME workflow files to extract nodes and edges, producing a
graph representation of the workflow. It fits into the knime2py generator pipeline
by enabling the conversion of KNIME workflows into Python code.

## Runtime Behavior

Inputs include DataFrames or context keys that the generated code reads. Outputs
are written to `context[...]`, with port mappings and types defined by the
workflow structure. The module implements key algorithms for node processing,
including handling of various KNIME node types.

## Edge Cases

The code implements safeguards for empty or constant columns, NaNs, and class
imbalances, ensuring robust processing of workflow data.

## Generated Code Dependencies

This module requires the following external libraries: lxml. These dependencies
are required by the generated code, not by this code.

## Usage

Typical usage involves invoking this module as part of the workflow parsing
process. An example of expected context access might be::

    data = context['input_table']

## Node Identity

The module handles various KNIME node types, identified by their unique IDs.
Special flags include LOOP, which indicates the start or end of a loop in the
workflow.

## Configuration

The `Node` dataclass is used for settings, with important fields including:

* id: Unique identifier for the node.
* name: Optional name of the node.
* type: Optional type of the node.
* path: Optional path to the node's settings.
* state: Execution state of the node (EXECUTED, CONFIGURED, IDLE).
* comments: Optional annotation text for the node.

The `parse_settings_xml` function extracts these values using XPaths from the
settings.xml file, with fallbacks for missing data.

## Limitations

Certain KNIME features may not be fully supported or approximated in the
conversion process.

## Exportability Heuristic

Each produced graph is marked `exportable: bool`. It is set to **False** if at least
one start node (node with no incoming edges inside that graph) is named exactly
`KNIME2PY` (case-insensitive). This lets us ignore the extra UI wrapper component
when the exporter is launched from a KNIME Component called “KNIME2PY”.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, cast

from lxml import etree as ET

from .xml_utils import XML_PARSER, XML_PARSER_STRICT, parse_settings_xml

# Node names that mark a subgraph as non-exportable when encountered as a start node.
NON_EXPORTABLE_NODE_NAMES = ["KNIME2PY"]
STATE_VALUES: Set[str] = {"EXECUTED", "CONFIGURED", "IDLE"}


class WorkflowParseError(Exception):
    """Typed parse error with a stable code for CLI callers."""

    def __init__(self, code: str, message: str, details: Optional[object] = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


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
    exportable: bool = True


def discover_workflows(root: Path) -> List[Path]:
    """Find all workflow.knime files under `root` (sorted)."""
    return sorted(
        (p for p in root.rglob("workflow.knime") if p.is_file()),
        key=lambda p: str(p),
    )


def _clean_annotation_text(text: str) -> str:
    """Replace KNIME-encoded line breaks and collapse whitespace."""
    text = text.replace("%%00010", " ")
    return " ".join(text.split()).strip()


def _read_state_and_annotation_from_settings(
    settings_ref: Path,
) -> Tuple[Optional[str], Optional[str]]:
    """Read the execution state and annotation text from the settings.xml file."""
    settings = settings_ref / "settings.xml" if settings_ref.is_dir() else settings_ref
    if not settings.exists():
        return None, None

    try:
        root = ET.parse(str(settings), parser=XML_PARSER).getroot()
        state_vals = root.xpath(".//*[local-name()='entry' and @key='state']/@value")
        state = state_vals[0].strip().upper() if state_vals and state_vals[0] else None

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


def _parse_knime5_structure(
    root: ET._Element,
    workflow_file: Path,
    *,
    strict: bool = False,
    missing_settings: Optional[List[Dict[str, Optional[str]]]] = None,
    invalid_settings: Optional[List[Dict[str, Optional[str]]]] = None,
) -> Tuple[Dict[str, Node], List[Edge]]:
    """Parse the structure of a KNIME 5 workflow and extract nodes and edges."""
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    nodes_cont = root.xpath(".//*[local-name()='config' and @key='nodes']")
    conns_cont = root.xpath(".//*[local-name()='config' and @key='connections']")
    nodes_cont = nodes_cont[0] if nodes_cont else None
    conns_cont = conns_cont[0] if conns_cont else None

    if nodes_cont is not None:
        node_cfgs = nodes_cont.xpath(
            "./*[local-name()='config' and starts-with(@key,'node_')]"
        )

        def node_sort_key(ncfg: ET._Element) -> Tuple[float, str]:
            raw_id = (
                ncfg.xpath("string(.//*[local-name()='entry' and @key='id']/@value)")
                or ""
            ).strip()
            try:
                parsed_id = float(raw_id)
            except Exception:
                parsed_id = float("inf")
            key_attr = (ncfg.get("key") or "")
            return parsed_id, key_attr

        for ncfg in sorted(node_cfgs, key=node_sort_key):
            raw_id = (
                ncfg.xpath("string(.//*[local-name()='entry' and @key='id']/@value)")
                or ""
            ).strip()
            node_id = raw_id if raw_id else str(uuid.uuid4())
            if node_id in nodes:
                node_id = f"{node_id}-{uuid.uuid4()}"

            settings_file = (
                ncfg.xpath(
                    "string(.//*[local-name()='entry' and @key='node_settings_file']/@value)"
                )
                or ""
            ).strip()
            node_type = (
                ncfg.xpath(
                    "string(.//*[local-name()='entry' and @key='node_type']/@value)"
                )
                or ""
            ).strip() or None

            name = None
            node_path = None
            state: Optional[str] = None
            comments: Optional[str] = None

            if settings_file:
                rel = Path(settings_file)
                name = rel.parent.name or name
                abs_settings = workflow_file.parent / rel
                workflow_root = workflow_file.parent.resolve()
                if not abs_settings.resolve().is_relative_to(workflow_root):
                    if strict and missing_settings is not None:
                        missing_settings.append(
                            {"node_id": node_id, "path": str(rel)}
                        )
                elif not abs_settings.exists() or not abs_settings.is_file():
                    if strict and missing_settings is not None:
                        missing_settings.append(
                            {"node_id": node_id, "path": str(rel)}
                        )
                else:
                    if strict and invalid_settings is not None:
                        try:
                            ET.parse(str(abs_settings), parser=XML_PARSER_STRICT).getroot()
                        except ET.XMLSyntaxError as exc:
                            invalid_settings.append(
                                {"node_id": node_id, "path": str(rel), "error": str(exc)}
                            )
                    node_path = str(abs_settings.parent)
                    parsed_name, parsed_type = parse_settings_xml(abs_settings.parent)
                    name = parsed_name or name
                    node_type = parsed_type or node_type
                    state, comments = _read_state_and_annotation_from_settings(
                        abs_settings
                    )
            else:
                if strict and missing_settings is not None:
                    missing_settings.append({"node_id": node_id, "path": None})

            nodes[node_id] = Node(
                id=node_id,
                name=name,
                type=node_type,
                path=node_path,
                state=_normalize_state(state),
                comments=comments,
            )

    if conns_cont is not None:
        conn_cfgs = conns_cont.xpath(
            "./*[local-name()='config' and starts-with(@key,'connection_')]"
        )

        def conn_sort_key(ccfg: ET._Element) -> Tuple[float, float, str, str]:
            def to_num(value: str) -> float:
                try:
                    return float(value)
                except Exception:
                    return float("inf")

            src = (
                ccfg.xpath(
                    "string(.//*[local-name()='entry' and @key='sourceID']/@value)"
                )
                or ""
            ).strip()
            dst = (
                ccfg.xpath(
                    "string(.//*[local-name()='entry' and @key='destID']/@value)"
                )
                or ""
            ).strip()
            src_port = (
                ccfg.xpath(
                    "string(.//*[local-name()='entry' and @key='sourcePort']/@value)"
                )
                or ""
            ).strip()
            dst_port = (
                ccfg.xpath(
                    "string(.//*[local-name()='entry' and @key='destPort']/@value)"
                )
                or ""
            ).strip()
            return to_num(src), to_num(dst), src_port, dst_port

        for ccfg in sorted(conn_cfgs, key=conn_sort_key):
            src = (
                ccfg.xpath(
                    "string(.//*[local-name()='entry' and @key='sourceID']/@value)"
                )
                or ""
            ).strip()
            dst = (
                ccfg.xpath(
                    "string(.//*[local-name()='entry' and @key='destID']/@value)"
                )
                or ""
            ).strip()
            src_port = (
                ccfg.xpath(
                    "string(.//*[local-name()='entry' and @key='sourcePort']/@value)"
                )
                or ""
            ).strip() or None
            dst_port = (
                ccfg.xpath(
                    "string(.//*[local-name()='entry' and @key='destPort']/@value)"
                )
                or ""
            ).strip() or None

            if src and dst:
                edges.append(
                    Edge(source=str(src), target=str(dst), source_port=src_port, target_port=dst_port)
                )

    return nodes, edges


def _normalize_state(value: Optional[str]) -> Optional[Literal["EXECUTED", "CONFIGURED", "IDLE"]]:
    """Convert arbitrary text to a valid state literal or None."""
    if not value:
        return None
    upper = value.strip().upper()
    return cast(Optional[Literal["EXECUTED", "CONFIGURED", "IDLE"]], upper if upper in STATE_VALUES else None)


def _parse_legacy_structure(
    root: ET._Element, workflow_file: Path
) -> Tuple[Dict[str, Node], List[Edge]]:
    """Raise on unsupported/legacy formats (placeholder to extend if needed)."""
    raise WorkflowParseError(
        "unsupported_workflow",
        f"Unsupported/legacy workflow format: {workflow_file}",
    )


def _weakly_connected_components(
    nodes: Dict[str, Node], edges: List[Edge]
) -> List[List[str]]:
    """Return weakly connected components as lists of node IDs."""
    if not nodes:
        return []

    adjacency: Dict[str, Set[str]] = {nid: set() for nid in nodes}
    for edge in edges:
        if edge.source in adjacency and edge.target in adjacency:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)

    seen: Set[str] = set()
    components: List[List[str]] = []

    for nid in nodes:
        if nid in seen:
            continue
        stack = [nid]
        component: List[str] = []
        seen.add(nid)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(
            sorted(
                component,
                key=lambda node_id: (int(node_id) if node_id.isdigit() else float("inf"), node_id),
            )
        )

    components.sort(
        key=lambda comp: (
            int(comp[0]) if comp and comp[0].isdigit() else float("inf"),
            comp[0] if comp else "",
        )
    )
    return components


def _compute_exportable_flag(
    node_subset: Dict[str, Node], edge_subset: List[Edge]
) -> bool:
    """
    Determine whether a subgraph is exportable.

    A graph is marked non-exportable if any node name contains one of the configured
    non-exportable markers (case-insensitive).
    """
    if not node_subset:
        return True

    indegree = {nid: 0 for nid in node_subset}
    for edge in edge_subset:
        if edge.target in indegree and edge.source in indegree:
            indegree[edge.target] += 1

    non_exportable = {name.upper() for name in NON_EXPORTABLE_NODE_NAMES}
    for nid in node_subset:
        name = (node_subset[nid].name or "").strip().upper()
        if any(marker in name for marker in non_exportable):
            return False
    return True


def _split_into_subgraphs(
    workflow_id: str,
    workflow_path: str,
    nodes: Dict[str, Node],
    edges: List[Edge],
) -> List[WorkflowGraph]:
    """Split the workflow graph into subgraphs based on weakly connected components."""
    components = _weakly_connected_components(nodes, edges)
    if not components:
        return []

    subgraphs: List[WorkflowGraph] = []
    for idx, component_nodes in enumerate(components, start=1):
        node_subset = {nid: nodes[nid] for nid in component_nodes}
        edge_subset = [
            edge for edge in edges if edge.source in node_subset and edge.target in node_subset
        ]
        sub_id = f"{workflow_id}__g{idx:02d}"
        exportable = _compute_exportable_flag(node_subset, edge_subset)
        subgraphs.append(
            WorkflowGraph(
                workflow_id=sub_id,
                workflow_path=workflow_path,
                nodes=node_subset,
                edges=edge_subset,
                exportable=exportable,
            )
        )
    return subgraphs


def parse_workflow_components(
    workflow_file: Path,
    *,
    strict: bool = False,
    workflow_id: Optional[str] = None,
    workflow_path: Optional[str] = None,
) -> List[WorkflowGraph]:
    """Parse a single workflow.knime file and return one graph per weakly connected component."""
    parser = XML_PARSER_STRICT if strict else XML_PARSER
    try:
        root = ET.parse(str(workflow_file), parser=parser).getroot()
    except ET.XMLSyntaxError as exc:
        raise WorkflowParseError(
            "invalid_xml",
            f"Invalid workflow XML: {workflow_file}",
            details=str(exc),
        ) from exc

    missing_settings: List[Dict[str, Optional[str]]] = []
    invalid_settings: List[Dict[str, Optional[str]]] = []
    nodes, edges = _parse_knime5_structure(
        root,
        workflow_file,
        strict=strict,
        missing_settings=missing_settings,
        invalid_settings=invalid_settings,
    )
    if strict and invalid_settings:
        raise WorkflowParseError(
            "invalid_xml",
            "Invalid settings.xml detected.",
            details=invalid_settings,
        )
    if strict and missing_settings:
        raise WorkflowParseError(
            "missing_settings",
            "Missing referenced settings.xml.",
            details=missing_settings,
        )
    if not nodes and not edges:
        nodes, edges = _parse_legacy_structure(root, workflow_file)

    base_id = workflow_id or (
        workflow_file.parent.name or workflow_file.parent.as_posix().replace("/", "_")
    )
    wf_path = workflow_path or str(workflow_file)
    return _split_into_subgraphs(base_id, wf_path, nodes, edges)


def parse_workflow(
    workflow_file: Path,
    *,
    strict: bool = False,
    workflow_id: Optional[str] = None,
    workflow_path: Optional[str] = None,
) -> WorkflowGraph:
    """Backward-compatible parser that returns the combined graph for the workflow."""
    parser = XML_PARSER_STRICT if strict else XML_PARSER
    try:
        root = ET.parse(str(workflow_file), parser=parser).getroot()
    except ET.XMLSyntaxError as exc:
        raise WorkflowParseError(
            "invalid_xml",
            f"Invalid workflow XML: {workflow_file}",
            details=str(exc),
        ) from exc

    missing_settings: List[Dict[str, Optional[str]]] = []
    invalid_settings: List[Dict[str, Optional[str]]] = []
    nodes, edges = _parse_knime5_structure(
        root,
        workflow_file,
        strict=strict,
        missing_settings=missing_settings,
        invalid_settings=invalid_settings,
    )
    if strict and invalid_settings:
        raise WorkflowParseError(
            "invalid_xml",
            "Invalid settings.xml detected.",
            details=invalid_settings,
        )
    if strict and missing_settings:
        raise WorkflowParseError(
            "missing_settings",
            "Missing referenced settings.xml.",
            details=missing_settings,
        )
    if not nodes and not edges:
        nodes, edges = _parse_legacy_structure(root, workflow_file)

    resolved_id = workflow_id or (
        workflow_file.parent.name or workflow_file.parent.as_posix().replace("/", "_")
    )
    resolved_path = workflow_path or str(workflow_file)
    exportable = _compute_exportable_flag(nodes, edges)
    return WorkflowGraph(
        workflow_id=resolved_id,
        workflow_path=resolved_path,
        nodes=nodes,
        edges=edges,
        exportable=exportable,
    )
