"""
knime2py — KNIME → Python workbook/code generator.

Overview
--------
`knime2py` parses a KNIME workflow (`workflow.knime`), reconstructs nodes and edges,
and emits runnable Python “workbooks” (Jupyter notebooks or scripts) together with
a machine-readable graph (JSON) and a Graphviz DOT file. Implemented KNIME nodes are
translated into idiomatic pandas / scikit-learn code via a pluggable registry.

Public API
----------
- ``parse_knime(path: str | Path)``: Parse a workflow directory or `workflow.knime`.
- ``emit_workbooks(...)``: Generate `.ipynb` / `.py` workbooks per component graph.
- ``traverse.*``: Utilities for graph extraction and deterministic ordering.
- ``nodes.*``: Per-node generators (handlers) producing Python code from `settings.xml`.

CLI
---
The console entry point is ``k2p`` (``python -m knime2py``). See ``k2p --help``.

Subpackages
-----------
- ``nodes``: Handlers for specific KNIME factories.
- ``parse_knime`` / ``traverse``: XML parsing and graph building.
- ``emitters``: Code and notebook emitters.

Notes
-----
Generated code depends on pandas / scikit-learn (and sometimes imbalanced-learn);
the generator itself depends on lxml for parsing. See the README for details.
"""

__all__ = ["parse_knime"]