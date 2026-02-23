# AGENTS.md

## Project Overview
- `knime2py` converts KNIME workflows into runnable Python and/or Jupyter workbooks.
- Core flow: parse workflow graph -> traverse nodes in deterministic order -> dispatch node handlers -> emit `.py`/`.ipynb` plus `.json`/`.dot`.
- Source package uses `src/` layout: `src/knime2py`.

## Important Paths
- `src/knime2py/cli.py`: CLI entrypoint (`k2p`, `knime2py`).
- `src/knime2py/parse_knime.py`: workflow and edge/node parsing.
- `src/knime2py/traverse.py`: deterministic traversal order.
- `src/knime2py/emitters.py`: graph/workbook writers.
- `src/knime2py/nodes/`: KNIME node exporters.
- `src/knime2py/nodes/registry.py`: dynamic handler discovery via `FACTORY`/`FACTORIES`.
- `tests/`: pytest suite and golden workflow fixtures.
- `tests/data/`: KNIME fixture projects and expected CSV outputs.
- `docs/`: MkDocs content.
- `rag/`: repository RAG indexing/query tools.

## Environment and Setup
- Python requirement from project metadata: `>=3.10` (project is typically used/tested on 3.11).
- Fast setup:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `./scripts/install.sh`
- `scripts/install.sh` defaults to installing `.[dev]`, `.[rag]`, and `.[ml-examples]`.
  - Toggle with env vars: `WITH_DEV`, `WITH_RAG`, `WITH_ML_EXAMPLES`, `CREATE_VENV`.

## Canonical Commands
- CLI help:
  - `python -m knime2py --help`
  - `k2p --help`
- Convert a sample workflow:
  - `python -m knime2py tests/data/KNIME_single_csv --out output --graph off --workbook py`
- Run tests:
  - `pytest -q`
  - `pytest -q tests/test_nodes_registry_handlers.py`
- Serve docs:
  - `mkdocs serve`
- Rebuild RAG index (optional, OpenAI key required for OpenAI embeddings):
  - `./scripts/rag_index_rebuild.sh`

## Agent Workflow Guidance
- Prefer focused edits in `src/knime2py/**` and targeted tests over full-suite runs while iterating.
- When adding/changing a KNIME node implementation:
  - Update/add module under `src/knime2py/nodes/`.
  - Ensure `FACTORY`/`FACTORIES` and `handle(...)` are correct so registry discovery works.
  - Add or update tests under `tests/test_nodes_*` or functional workflow tests.
- Preserve stable CLI error contract (JSON error envelope + exit codes) in `cli.py`.
- Keep output filenames and graph/workbook naming deterministic.

## Repo-Specific Cautions
- Do not casually edit large fixture trees in `tests/data/**` unless the task is fixture-related.
- `tests/data/!output`, `output/`, and `.rag_index/` are generated artifacts and may be safely regenerated.
- Some helper scripts are intentionally opinionated and destructive:
  - `scripts/k2p.sh`, `scripts/k2p_pex.sh`, `scripts/k2p_docker.sh` clear output directories.
  - `scripts/k2p.sh` currently contains a hardcoded local workflow path near the end; adjust before use.
- `scripts/release.sh` requires a clean git tree and pushes commit + tag.

## Validation Expectations for Changes
- Minimum for code edits:
  - Run targeted pytest files that cover touched modules.
  - Run at least one CLI invocation for smoke coverage.
- For parser/emitter/registry changes:
  - Prefer running the relevant functional workflow tests in `tests/test_functional_*` or `tests/test_*workflow*`.
- If tests are skipped due to time or environment, explicitly report what was not run.
