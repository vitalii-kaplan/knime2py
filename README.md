# knime2py 
# KNIME → Python Workbook

A command-line tool that parses a KNIME project and emits:

* a “Python workbook” for each discovered workflow as either a **Jupyter notebook** (`*_workbook.ipynb`) or a **Python script** (`*_workbook.py`)
* a machine-readable graph with KNIME nodes (`<workflow>.json`)
* a Graphviz DOT file (`<workflow>.dot`)

> Status: prototype. Can parse KNIME project, create graphs and Python workbook with one empty section per node. 
---

## Features

* **Workflow discovery** — recursively finds all `workflow.knime` files under a root.
* **Parser (KNIME 5.x)**
  * KNIME 5.x: nodes under `<config key="nodes">/config key="node_*">`; edges under `<config key="connections">/config key="connection_*">`.
  * Legacy: Unsupported.
* **Node enrichment** — reads each node’s `settings.xml` when present to get a human label and factory class (type).
* **Outputs per workflow**

  * `*.json` graph (nodes, edges, metadata)
  * `*.dot` Graphviz (left-to-right)
  * `*_workbook.ipynb` (default) or `*_workbook.py` with one section per node
* **Topological ordering** — workbook sections ordered by DAG topological sort; if a cycle is detected, remaining nodes are appended in a stable order.

---

## Requirements

* Python 3.8+
* (Optional) Graphviz (`dot`) to render `.dot` files
* A KNIME project directory containing one or more `workflow.knime` files

---

## Quick start

```bash
# Run against a KNIME project root, write outputs to ./out
python k2p.py /path/to/KNIME_project --out out

# Generate a Jupyter notebook workbook (default)
python k2p.py /path/to/KNIME_project --out out --workbook ipynb

# Generate a Python script workbook
python k2p.py /path/to/KNIME_project --out out --workbook py

# Generate both formats
python k2p.py /path/to/KNIME_project --out out --workbook both

# Only top-level workflows (skip nested ones with deeper paths)
python k2p.py /path/to/KNIME_project --out out --toponly
```

Outputs are placed in the `--out` directory, one set per discovered workflow.

---

## CLI

```
usage: k2p.py [-h] [--out OUT] [--toponly] [--workbook {py,ipynb,both}] root

positional arguments:
  root                  Path to KNIME project root directory

options:
  -h, --help            Show help message and exit
  --out OUT             Output directory for JSON/DOT/Workbook (default: out_graphs)
  --toponly             Emit only the shallowest (top-level) workflows by path depth
  --workbook {py,ipynb,both}
                        Which workbook format(s) to generate (default: ipynb)
```

## KNIME compatibility

* **Supported**: KNIME 5.x exports where `workflow.knime` stores nodes under `<config key="nodes">` and connections under `<config key="connections">`.
* **Legacy**: Unsupported. 
* **Components / metanodes / loops**: Any discovered `workflow.knime` (including nested) is treated as a separate workflow. Detailed expansion of component semantics is **not** implemented yet.

If nodes do not appear for a workflow, provide a minimal example so XML paths can be extended.

---

## Design notes

* Namespace-tolerant parsing with fallbacks when attributes are missing.
* Node labels inferred from the directory portion of `node_settings_file` and refined from `settings.xml` (entries like `name`, `label`, `factory`).
* Topological order drives workbook section order. On cycles, remaining nodes are appended in stable order.

---

## Contributing

* Open an issue with a small sample workflow (redacted is fine) and your KNIME version.
* PRs for new node translators should include: a tiny test workflow, emitted code stub, and brief notes on assumptions.

---

## License
MIT

---

## Acknowledgements

KNIME® is a trademark of KNIME AG. This project is an independent community effort and is not affiliated with KNIME AG.
