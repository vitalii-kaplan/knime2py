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

## Installation & Distribution

You can use knime2py in three ways.

### 1) Docker image (no local Python/pip needed)

```bash
docker pull ghcr.io/vitaly-chibrikov/knime2py:latest
docker run --rm ghcr.io/vitaly-chibrikov/knime2py:latest --help
````

Typical run:

```bash
docker run --rm \
  -v "$PWD/workflow":/wf:ro \
  -v "$PWD/out":/out \
  ghcr.io/vitaly-chibrikov/knime2py:latest \
  /wf --out /out --workbook both
```

Preserve host absolute paths (mirror the path inside the container):

```bash
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$PWD":"$PWD" -w "$PWD" \
  ghcr.io/vitaly-chibrikov/knime2py:latest \
  "$PWD/workflow" --out "$PWD/out" --graph off
```

A helper script is available: `k2p_docker.sh` (see repository root).

### 2) PEX single-file binaries (require Python **3.11** on the user’s machine)

Download from **Releases**. Example (macOS/Linux):

```bash
python3 --version          # must be 3.11.x
chmod +x k2p-macos-<arch>.pex
python3 k2p-macos-<arch>.pex --help

# Example
python3 k2p-macos-<arch>.pex /path/to/workflow --out /path/to/out --graph off
```

Windows (PowerShell):

```powershell
py -3.11 k2p-windows.pex --help
py -3.11 k2p-windows.pex C:\path\to\workflow --out C:\path\to\out --graph off
```

### 3) Source (developer) install

```bash
python -m pip install --upgrade pip
pip install -e .
# optional: run tests
pytest -q
```

---

## Quick start (CLI)

The console entrypoint is **`k2p`**. You can also use `python -m knime2py`.

```bash
# Generate BOTH notebook and script (omit --workbook)
k2p /path/to/workflow.knime --out out_dir

# Or pass a directory that contains exactly one workflow.knime
k2p /path/to/knime_project_dir --out out_dir

# Only notebook
k2p /path/to/workflow.knime --out out_dir --workbook ipynb

# Only script
k2p /path/to/workflow.knime --out out_dir --workbook py

# Disable graph files
k2p /path/to/workflow.knime --out out_dir --graph off
```

---

## What gets emitted

**Per isolated component** (weakly connected subgraph):

```
<base>__g01.json
<base>__g01.dot
<base>__g01_workbook.ipynb
<base>__g01_workbook.py
<base>__g02.json
…
```

Where `<base>` is the workflow directory name; `__gNN` is the component index.

* **Graph JSON** – node ids, ported edges, node names/types/paths.
* **Graphviz DOT** – left-to-right graph with labels; render with:

  ```bash
  dot -Tpng <base>__g01.dot -o component01.png
  ```
* **Workbooks**

  * **Notebook**: one markdown cell + one code cell per node; shared `context` dict.
  * **Script**: `node_<id>_<title>()` functions, a **single consolidated import block**, shared `context`, `run_all()`.

---

## CLI reference

```
usage: k2p [-h] [--out OUT] [--workbook {py,ipynb}] [--graph {dot,json,off}] path

positional arguments:
  path                  Path to a workflow.knime file OR a directory containing exactly one workflow.knime

options:
  -h, --help            Show help message and exit
  --out OUT             Output directory (default: out_graphs)
  --workbook {py,ipynb} Workbook format to generate. Omit to generate both.
  --graph {dot,json,off}
                        Which graph file(s) to emit. Omit to generate both; use "off" to skip.
```

---

## API reference

* Module documentation generated from docstrings: [**knime2py API**](api/knime2py/)

## Implemented node exporters

* List of all [`implemented nodes is here.`](src/knime2py/nodes/)

  Unsupported nodes produce a TODO stub with all parameters from `settings.xml` initialized.

---

## Reproducibility & randomness

Some KNIME nodes involve randomness (e.g., **Partitioning**, **Equal Size Sampling**). The generated Python uses pandas/scikit-learn RNGs. Even with the same seed, exact row identities can differ from KNIME’s RNG. Guarantees:

* **Class proportions and split sizes** match the requested settings.
* With identical inputs and seed, Python runs are reproducible **within Python**.
* Stratification preserves target distribution; infeasible cases fall back safely.

> Compare sizes and distributions, not exact row sets.

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


