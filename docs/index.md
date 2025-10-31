# knime2py — KNIME → Python Workbook

**knime2py** is a KNIME→Python exporter and code generator. It parses a KNIME workflow, reconstructs its nodes and connections, and emits runnable Python “workbooks” (Jupyter notebook and/or script) by translating supported KNIME nodes into idiomatic **pandas** / **scikit-learn** code via a pluggable node registry. In addition to executable code, it writes a machine-readable graph (**JSON**) and a **Graphviz DOT** file, preserving port wiring and execution order so the generated Python mirrors the original workflow.

---

## Features

- **Single-workflow focus** – pass either a `workflow.knime` file or a directory that *directly* contains `workflow.knime`.  
  Subdirectories are not traversed.
- **Components** – nested component workflows are supported if you point directly at the component’s folder (it also has its own `workflow.knime`).
- **Isolated graphs detection** – splits the workflow into **unconnected graphs**; each becomes its own output set with an ID suffix like `__g01`, `__g02`, …
- **Deterministic ordering** – a stable depth-first traversal ensures nodes run after all predecessors; cyclic/disconnected regions are handled with a consistent fallback order.
- **Pluggable node registry** – KNIME nodes are mapped to emitters in `knime2py.nodes.*`; unsupported nodes get a clear stub with initialized parameters from `settings.xml`.

---

## Requirements

- Python **3.9+** (project is developed/tested on **3.11**).
- Runtime libs used by **generated code**: `pandas`, `scikit-learn` (bundled in Docker/PEX; for source installs see below).
- Optional: Graphviz CLI (`dot`) if you want to render `.dot` files; the tool always writes `.dot`.

---

## Contributing

* Emitters live under `src/knime2py/nodes/`.
* Each emitter:

  * Parses `settings.xml`.
  * Generates imports and a code body that reads/writes the shared `context`.
  * Declares a `handle()` hook for registry lookup.
* Tests live under `tests/`; helpers under `tests/support/`.

---

## License

MIT

---

## Acknowledgements

KNIME® is a trademark of KNIME AG. This project is an independent community effort and is not affiliated with KNIME AG.


