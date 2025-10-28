# knime2py/nodes/registry.py
from __future__ import annotations

"""
This module discovers and registers KNIME node handlers.

Overview
----------------------------
This module imports all node definitions from the knime2py/nodes/ directory and maps
their factory IDs to the corresponding modules. It facilitates the dynamic loading of
node handlers based on their defined factories.

Runtime Behavior
----------------------------
Inputs:
- The generated code reads DataFrames or context keys as specified by the node's
  factory.

Outputs:
- The module writes to `context[...]`, mapping the output ports to the generated
  code's results.

Key algorithms:
- The module collects factory IDs from each node module, ensuring no duplicates
  while preserving the order of registration.

Edge Cases
----------------------------
The code handles cases where modules may not define a factory or where multiple
modules define the same factory ID, logging appropriate warnings.

Generated Code Dependencies
----------------------------
The generated code may depend on external libraries such as pandas, numpy, and
sklearn. These dependencies are required for the execution of the generated code,
not for this module itself.

Usage
----------------------------
This module is typically invoked by the knime2py emitter to dynamically load node
handlers. An example of expected context access might look like:
```python
result = context["output_port"]
```

Node Identity
----------------------------
KNIME factory IDs are defined in the node modules as FACTORY or FACTORIES constants.
The module supports a default handler for cases where no specific factory is defined.

Configuration
----------------------------
The settings for each node are typically defined in a `@dataclass`, which includes
fields such as `input_data` and `output_data`. The `parse_*` functions extract these
values from the node's settings.xml file.

Limitations
----------------------------
This module does not support all KNIME node features and may approximate behavior
in certain cases.

References
----------------------------
For more information, refer to the KNIME documentation and the HUB_URL constant.
"""

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

    Args:
        mod (ModuleType): The module from which to collect factory IDs.

    Returns:
        List[str]: A list of factory IDs collected from the module.
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
    """
    Check if the given module has a callable 'handle' function.

    Args:
        mod (ModuleType): The module to check.

    Returns:
        bool: True if the module has a callable 'handle' function, False otherwise.
    """
    return callable(getattr(mod, "handle", None))


def discover_handlers() -> Dict[str, ModuleType]:
    """
    Import every .py in knime2py/nodes/ and map FACTORY -> module.
    FACTORY == "" is allowed and denotes the default "not implemented" handler.
    Lower PRIORITY (int) wins on duplicates; ties by module name.

    Returns:
        Dict[str, ModuleType]: A dictionary mapping factory IDs to their corresponding modules.
    """
    candidates: List[tuple[int, str, str, ModuleType]] = []

    # --- Materialize the list of module specs first (helps debugging) ---
    try:
        specs = list(pkgutil.iter_modules(_nodes_pkg.__path__))
    except Exception as e:
        print(f"[nodes.registry] Failed to list node modules: {e}", file=sys.stderr)
        specs = []

    # Deterministic order while debugging
    specs.sort(key=lambda s: (int(s.ispkg), s.name))

    # --- Iterate over the collected specs ---
    for spec in specs:
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
            # No handle() function -> not a node handler
            continue

        factories = _iter_factories(mod)
        if not factories:
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
    """
    Return { FACTORY_ID: module }, including possibly mapping[''] for the default handler.

    Returns:
        Dict[str, ModuleType]: A dictionary mapping factory IDs to their corresponding modules.
    """
    global _HANDLERS_MAP
    if _HANDLERS_MAP is None:
        _HANDLERS_MAP = discover_handlers()
    return _HANDLERS_MAP


def get_default_handler() -> ModuleType | None:
    """
    Return the default handler module (FACTORY == ''), or None if not present.

    Returns:
        ModuleType | None: The default handler module or None if not found.
    """
    return get_handlers().get("")
