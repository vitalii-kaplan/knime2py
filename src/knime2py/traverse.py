#!/usr/bin/env python3
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
    Depth-biased traversal that *emits* a node only after all of its predecessors
    have been visited. It explores as deep as possible along outgoing edges, but
    will recursively visit unvisited predecessors first (backtracking as needed).

    - Deterministic: numeric node IDs first, then lexicographic.
    - Safe on cycles: nodes in the current recursion stack are not re-entered.
    - Covers disconnected components.
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


def traverse_nodes(g) -> Iterator[dict]:
    """
    Yield per-node context dicts in depth-ready order:
      {
        "nid": str,
        "node": Node,
        "title": str,
        "root_id": str,
        "state": Optional[str],
        "comments": Optional[str],
        "incoming": list[(src_id, Edge)],
        "outgoing": list[(dst_id, Edge)],
      }
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
