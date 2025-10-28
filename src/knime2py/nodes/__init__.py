"""KNIME → Python node emitters.

This package contains per-node generator modules that translate specific KNIME
node *factories* into executable Python code (pandas / scikit-learn) used by
**knime2py**. Use this section of the documentation to:

- verify whether a KNIME node is implemented (one module per factory);
- inspect how its code generation works and what limitations apply.

Conventions
-----------
- Each module exposes a ``FACTORY`` (or ``*_FACTORY``) string with the KNIME
  factory id and a public ``handle(ntype, nid, npath, incoming, outgoing)``
  entry point that returns ``(imports, body_lines)`` or ``None``.
- Many modules also define ``HUB_URL``, ``generate_imports()``,
  ``generate_py_body(...)``, and a settings ``@dataclass`` parsed from
  the node’s ``settings.xml``.

Fallback
--------
If no module exists for a factory, the fallback ``not_implemented`` handler
emits a stub with parameters initialized from ``settings.xml``.
"""

__all__: list[str] = []
