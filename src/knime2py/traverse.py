#!/usr/bin/env python3
"""
Traverse nodes in a directed graph and yield their context in depth-first order.

Overview
----------------------------
This module provides functionality to traverse nodes in a directed graph, 
emitting each node's context only after all of its predecessors have been 
visited. It fits into the knime2py generator pipeline by ensuring that 
nodes are processed in a logical order based on their dependencies.

Runtime Behavior
----------------------------
Inputs:
- The module reads a graph object containing nodes and edges, where nodes 
  are indexed by their IDs.

Outputs:
- The module yields dictionaries containing information about each node, 
  including its ID, the node object, title, root ID, state, comments, 
  and incoming/outgoing edges.

Key algorithms:
- The depth-first search (DFS) algorithm is used to traverse the graph, 
  ensuring that all predecessors are visited before a node is emitted. 
  The traversal is deterministic, sorting numeric IDs before non-numeric 
  ones.

Edge Cases
----------------------------
The code handles cycles in the graph by maintaining a recursion stack to 
prevent re-entering nodes. It also covers disconnected components by 
ensuring that all nodes are visited, even if they are not connected to 
the root nodes.

Generated Code Dependencies
----------------------------
The generated code may depend on external libraries such as pandas, 
numpy, and others, depending on the specific context and operations 
performed. These dependencies are required by the generated code, not 
by this module.

Usage
----------------------------
This module is typically invoked by the emitter as part of the 
knime2py workflow. An example of expected context access might be:
```python
context['node_id'] = node['nid']
```

Node Identity
----------------------------
This module generates code based on the settings defined in 
`settings.xml`. The KNIME factory IDs and any special flags are 
defined within the context of the nodes being processed.

Configuration
----------------------------
The settings are managed using a `@dataclass`, which includes fields 
that define the behavior of the nodes. The `parse_*` functions extract 
these values from the provided paths or xpaths, with appropriate 
fallbacks.

Limitations
----------------------------
Certain options available in KNIME may not be fully supported or may 
be approximated in the generated code. Users should be aware of these 
differences.

References
----------------------------
For more information, refer to the KNIME documentation and relevant 
terminology related to graph traversal and node processing.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple


__all__ = [
    "depth_order",
    "traverse_nodes",
    "derive_title_and_root"
]

def _id_sort_key(s: str) -> Tuple[int, str]:
    """Sort numeric IDs before non-numeric, keep numeric order stable."""
    return (0, int(s)) if s.isdigit() else (1, s)


def _build_edge_maps(edges: Iterable) -> Tuple[dict, dict]:
    """
    Build incoming/outgoing maps for quick neighbor lookups.
    
    Returns (incoming_map, outgoing_map) where:
      incoming_map[target_id] -> [Edge...]
      outgoing_map[source_id] -> [Edge...]
    
    Args:
        edges (Iterable): An iterable of edges with source and target attributes.

    Returns:
        Tuple[dict, dict]: A tuple containing the incoming and outgoing edge maps.
    """
    inc = defaultdict(list)
    out = defaultdict(list)
    for e in edges:
        # duck-typing: expect .source and .target attributes
        out[e.source].append(e)
        inc[e.target].append(e)
    return inc, out


def depth_order(nodes: Dict[str, object], edges: Iterable) -> List[str]:
    """
    Perform a depth-biased traversal of nodes, emitting each node only after 
    all of its predecessors have been visited.

    This function explores as deep as possible along outgoing edges, 
    recursively visiting unvisited predecessors first (backtracking as needed).

    - Deterministic: numeric node IDs first, then lexicographic.
    - Safe on cycles: nodes in the current recursion stack are not re-entered.
    - Covers disconnected components.

    Args:
        nodes (Dict[str, object]): A dictionary of nodes indexed by their IDs.
        edges (Iterable): An iterable of edges connecting the nodes.

    Returns:
        List[str]: A list of node IDs in the order they should be processed.
    """
    # Build predecessor/successor maps
    succ = defaultdict(list)   # u -> [v...]
    preds = defaultdict(set)   # v -> {u...}
    for e in edges:
        if e.source in nodes and e.target in nodes:
            succ[e.source].append(e.target)
            preds[e.target].add(e.source)

    # Ensure every node appears in maps
    for nid in nodes.keys():
        succ.setdefault(nid, [])
        preds.setdefault(nid, set())

    # Sort children deterministically
    for u in list(succ.keys()):
        succ[u].sort(key=_id_sort_key)

    # Roots: no predecessors
    roots = sorted([nid for nid in nodes if not preds[nid]], key=_id_sort_key)

    order: List[str] = []
    visited = set()
    onstack = set()  # cycle guard

    def dfs(u: str):
        """Depth-first search helper to visit nodes."""
        if u in visited or u in onstack:
            return
        onstack.add(u)

        # Ensure all predecessors are visited first (may recurse upstream)
        for p in sorted(preds[u], key=_id_sort_key):
            if p not in visited:
                dfs(p)

        # Now it's safe to emit u
        if u not in visited:
            visited.add(u)
            order.append(u)

        # Go deep along successors
        for v in succ[u]:
            if v not in visited:
                dfs(v)

        onstack.remove(u)

    # Traverse from roots
    for r in roots:
        dfs(r)

    # Cover leftovers (cycles / disconnected)
    for nid in sorted(nodes.keys(), key=_id_sort_key):
        if nid not in visited:
            dfs(nid)

    return order


def derive_title_and_root(nid: str, n) -> tuple[str, str]:
    """
    Derive the title and root ID from a node.

    This function attempts to extract the title and root ID from the node's 
    path if available, otherwise it falls back to the node's name or type.

    Args:
        nid (str): The ID of the node.
        n: The node object from which to derive the title and root ID.

    Returns:
        tuple[str, str]: A tuple containing the title and root ID.
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


def traverse_nodes(g) -> Iterator[dict]:
    """
    Yield per-node context dictionaries in depth-ready order.

    Each dictionary contains information about the node, including its ID, 
    the node object, title, root ID, state, comments, and incoming/outgoing edges.

    Args:
        g: A graph object containing nodes and edges.

    Yields:
        Iterator[dict]: A generator yielding dictionaries with node context.
    """
    order = depth_order(g.nodes, g.edges)
    incoming_map, outgoing_map = _build_edge_maps(g.edges)

    for nid in order:
        n = g.nodes[nid]
        title, root_id = derive_title_and_root(nid, n)
        state = getattr(n, "state", None)
        comments = getattr(n, "comments", None)

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
            "comments": comments,
            "incoming": incoming,
            "outgoing": outgoing,
        }

