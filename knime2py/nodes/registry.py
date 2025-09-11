# knime2py/nodes/registry.py
from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType
from typing import List

# Package we’ll scan
from knime2py import nodes as _nodes_pkg

# Optional: modules to skip even if present in nodes/
_SKIP = {
    "node_utils",
}

def _is_handler_module(mod: ModuleType) -> bool:
    """A handler module must expose callables: can_handle(node_type) and handle(...)."""
    can = getattr(mod, "can_handle", None)
    handle = getattr(mod, "handle", None)
    return callable(can) and callable(handle)

def discover_handlers() -> List[ModuleType]:
    """
    Import every .py in knime2py/nodes/ and keep those exposing can_handle() + handle().
    Handlers may optionally define PRIORITY (int, lower loads first). Default=100.
    """
    found = []
    for spec in pkgutil.iter_modules(_nodes_pkg.__path__):
        if spec.ispkg:
            continue
        name = spec.name
        if name.startswith("_") or name in _SKIP:
            continue
        try:
            mod = importlib.import_module(f"{_nodes_pkg.__name__}.{name}")
        except Exception as e:
            print(f"[nodes.registry] Skipping {name}: import error: {e}", file=sys.stderr)
            continue
        if _is_handler_module(mod):
            prio = getattr(mod, "PRIORITY", 100)
            found.append((prio, name, mod))
    # stable order by priority then name
    found.sort(key=lambda x: (x[0], x[1]))
    return [mod for _, _, mod in found]

# Simple cache
_HANDLERS: List[ModuleType] | None = None

def get_handlers() -> List[ModuleType]:
    global _HANDLERS
    if _HANDLERS is None:
        _HANDLERS = discover_handlers()
    return _HANDLERS
