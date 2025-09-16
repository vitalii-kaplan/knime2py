# knime2py/nodes/registry.py
from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType
from typing import Dict, List

from knime2py import nodes as _nodes_pkg

_SKIP = {"node_utils"}


def _iter_factories(mod: ModuleType) -> List[str]:
    """
    Collect factory IDs from a module. Supports:
      - FACTORY: str (may be empty "" to mean 'default handler')
      - FACTORIES: iterable[str]
    Empty string ("") is preserved to represent the default handler.
    """
    vals: List[str] = []

    if hasattr(mod, "FACTORY"):
        f = getattr(mod, "FACTORY")
        if isinstance(f, str):
            vals.append(f.strip())  # may be ""
        elif isinstance(f, (list, tuple, set)):
            for x in f:
                if isinstance(x, str):
                    vals.append(x.strip())

    if hasattr(mod, "FACTORIES"):
        f2 = getattr(mod, "FACTORIES")
        if isinstance(f2, str):
            vals.append(f2.strip())
        elif isinstance(f2, (list, tuple, set)):
            for x in f2:
                if isinstance(x, str):
                    vals.append(x.strip())

    # de-duplicate while preserving order
    out: List[str] = []
    seen = set()
    for s in vals:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _has_handle(mod: ModuleType) -> bool:
    return callable(getattr(mod, "handle", None))


def discover_handlers() -> Dict[str, ModuleType]:
    """
    Import every .py in knime2py/nodes/ and map FACTORY -> module.
    FACTORY == "" is allowed and denotes the default "not implemented" handler.
    Lower PRIORITY (int) wins on duplicates; ties by module name.
    """
    candidates: List[tuple[int, str, str, ModuleType]] = []

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

        if not _has_handle(mod):
            continue

        factories = _iter_factories(mod)
        if factories is None:
            continue

        prio = getattr(mod, "PRIORITY", 100)
        for fac in factories:
            candidates.append((prio, name, fac, mod))

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))

    mapping: Dict[str, ModuleType] = {}
    claimed_by: Dict[str, str] = {}

    for prio, name, fac, mod in candidates:
        if fac in mapping:
            prev = claimed_by[fac]
            print(
                f"[nodes.registry] Duplicate FACTORY '{fac}' in module '{name}' ignored; "
                f"already provided by '{prev}'.",
                file=sys.stderr,
            )
            continue
        mapping[fac] = mod
        claimed_by[fac] = name

    return mapping


_HANDLERS_MAP: Dict[str, ModuleType] | None = None


def get_handlers() -> Dict[str, ModuleType]:
    """Return { FACTORY_ID: module }, including possibly mapping[''] for the default handler."""
    global _HANDLERS_MAP
    if _HANDLERS_MAP is None:
        _HANDLERS_MAP = discover_handlers()
    return _HANDLERS_MAP


def get_default_handler() -> ModuleType | None:
    """Return the default handler module (FACTORY == ''), or None if not present."""
    return get_handlers().get("")
