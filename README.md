# knime2py — KNIME → Python Workbook

knime2py is a code-generation and KNIME→Python exporter: it parses a KNIME workflow, reconstructs its nodes and connections, and emits runnable Python “workbooks” (Jupyter notebook or script) by translating supported KNIME nodes into idiomatic pandas / scikit-learn code via a pluggable node registry. Alongside the executable code, it also writes a machine-readable graph (JSON) and a Graphviz DOT file, preserving port wiring and execution order so the generated Python mirrors the original workflow.

The project includes command-line tool that parses a **KNIME workflow** and emits, for each isolated subgraph (component) inside it:

* a machine-readable graph (`<workflow_id>__gNN.json`)
* a Graphviz DOT file (`<workflow_id>__gNN.dot`)
* a “Python workbook” as either a **Jupyter notebook** (`<workflow_id>__gNN_workbook.ipynb`) or a **Python script** (`<workflow_id>__gNN_workbook.py`) — if you omit `--workbook`, both are generated

> Status: prototype/MVP. KNIME 5.x workflows supported. Legacy (<node>/<connection>) not supported.

---

## Features

* **Single-workflow focus** — point at a `workflow.knime` or a directory containing exactly one `workflow.knime`.
* **Isolated graphs detection** — splits the workflow into **unconnected graphs**; each becomes its own output set with an ID suffix like `__g01`, `__g02`, …
* **Depth-ready ordering** — sections are emitted by a deterministic depth-first traversal that only visits a node once all of its predecessors have been visited; in cyclic or disconnected regions it continues depth-first and then appends any remaining nodes in a stable order.

---

## Requirements

* Python 3.8+
* [`lxml`](https://lxml.de/) (XML parsing)
* [`pandas`](https://pandas.pydata.org/) (generated code uses it)
* [`scikit-learn`](https://scikit-learn.org/) (used by some nodes, e.g., Partitioning, Equal Size Sampling)
* (Optional) Graphviz CLI to render `.dot` (`dot`, `neato`, etc.)

---

## Quick start

```bash
# From the repo root:
# Generate BOTH notebook and script (default when --workbook is omitted)
python k2p.py /path/to/workflow.knime --out out_dir

# Or pass a directory that contains exactly one workflow.knime
python k2p.py /path/to/knime_project_dir --out out_dir

# Only notebook
python k2p.py /path/to/workflow.knime --out out_dir --workbook ipynb

# Only script
python k2p.py /path/to/workflow.knime --out out_dir --workbook py
```

Outputs are written to `out_dir/` with one set **per component**:

```
<base>__g01.json
<base>__g01.dot
<base>__g01_workbook.ipynb
<base>__g01_workbook.py
<base>__g02.json
…
```

`<base>` is the workflow directory name; `__gNN` is the component index.

---

## CLI

```
usage: k2p.py [-h] [--out OUT] [--workbook {py,ipynb}] path

positional arguments:
  path                  Path to a workflow.knime file OR a directory containing exactly one workflow.knime

options:
  -h, --help            Show help message and exit
  --out OUT             Output directory (default: out_graphs)
  --workbook {py,ipynb}
                        Workbook format to generate. Omit to generate both.
```

---

## What gets emitted

### Graph JSON (per component)

* Nodes keyed by KNIME node id (strings)
* Edges with `source`, `target`, and optional `source_port` / `target_port`
* Node `name`, `type` (factory class), and `path` when discoverable

### Graphviz DOT (per component)

Left-to-right graph with node labels:

```dot
digraph knime {
  rankdir=LR;
  "1" [shape=box, style=rounded, label="1\nCSV Reader\n<org.knime...CSVTableReaderNodeFactory>"];
  "2" [shape=box, style=rounded, label="2\nCSV Writer\n<org.knime...CSVWriter2NodeFactory>"];
  "1" -> "2" [taillabel="1", headlabel="1"];
}
```

Render example:

```bash
dot -Tpng <base>__g01.dot -o component01.png
```

### Workbooks (per component)

**Notebook (`.ipynb`)**
For each node, a markdown cell (title + port summaries) followed by a code cell that references a shared `context` dict.

**Script (`.py`)**
Functions named `node_<id>_<title>()` with the same metadata embedded as comments, a **single import preamble**, a shared `context` dict, and a `run_all()` that calls nodes in topological order.

---

## Implemented node generators

List of all implemented nodes is here: https://vitaly-chibrikov.github.io/knime2py/implemented.html

> If a node type isn’t implemented, a clear TODO stub is emitted with all paramaters from node's settings.xml file initialized.

---

## Reproducibility & randomness

Some KNIME nodes involve randomness (e.g., **Partitioning**, **Equal Size Sampling**). Our generated Python uses **pandas** and **scikit-learn** RNGs. Even when the same seed value is used, **row-for-row results may differ from KNIME’s internal RNG**. What we do guarantee:

* **Class proportions and train/test sizes** match the requested settings.
* With the same inputs and seed, Python runs are reproducible within Python.
* Stratified operations preserve target distribution; when stratification is infeasible (e.g., minuscule classes), a guarded fallback is used.

In other words, **statistics match, but exact row identities may not**. Tests should compare **sizes and distributions**, not exact row sets.

---

## License

MIT

---

## Acknowledgements

KNIME® is a trademark of KNIME AG. This project is an independent community effort and is not affiliated with KNIME AG.