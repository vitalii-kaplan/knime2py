# RAG for `knime2py` — Design & Usage

This document describes the Retrieval-Augmented Generation (RAG) layer implemented for the project. It covers the architecture, utilities, scripts, configuration, prompt/retrieval strategy, token budgeting, and concrete usage examples.

---

## 1) High-level Overview

1. **Goal.** Answer repository questions and safely rewrite single files using an LLM, grounding the LLM on the codebase via a local **ChromaDB** index.
2. **Backends.**
    1. **Embeddings:** `openai` (e.g., `text-embedding-3-large`) or **local SBERT** (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
    2. **Generation:** **OpenAI** chat models (`gpt-4o*`, etc.) or **Ollama** (local models like `llama3`).
3. **Index.** A persistent Chroma index under `./.rag_index/` with:
    1. a **chunk collection** for content passages, and
    2. a **manifest collection** mapping basenames (e.g., `registry.py`) to full repo paths, used for filename-hint retrieval.
4. **Repository structure file.** A small `rag/.generated/STRUCTURE.md` is assumed to exist and is used to give the LLM top-level orientation; a small number of its chunks are always injected (reserved slots).

---

## 2) Implemented Components

### 2.1 `rag/rag_utils.py` (shared utilities)

1. **Config & Naming**
    1. `RAGConfig`, `load_config_from_env(...)`
    2. Collection naming: `current_collection_name(...)`, `manifest_collection_name(...)`
2. **Chroma Access**
    1. `get_client(...)`, `get_collection(...)`, `get_manifest(...)`
3. **Embeddings**
    1. `encode_query(...)` (OpenAI or SBERT)
4. **Retrieval Primitives**
    1. `retrieve_raw(...)`, `fetch_file_chunks_by_path(...)`
    2. Filename hints: `extract_file_hints(...)`, `find_paths_by_basenames(...)`
    3. Structure slice: `structure_chunks(...)`
5. **Composite Retrieval**
    1. `retrieve_with_structure_and_hints(...)` — order without rerank:
        1. filename-hinted chunks (per-file cap),
        2. `STRUCTURE.md` chunks (reserved small slice),
        3. vector search fill;
        4. optional rerank via cross-encoder (`RAG_RERANK=1`).
6. **Prompt Utilities**
    1. `format_chunks(...)` (for QA),
    2. `build_qa_prompt(system_prompt, question, passages, extra_instructions="")`,
    3. `QA_SYSTEM_PROMPT` and `system_prompt(...)` (overrides via `RAG_SYS_PROMPT` or `RAG_SYS_PROMPT_FILE`)
7. **Token Budgeting**
    1. `count_tokens(...)`, `ensure_prompt_fits(...)`, `resolve_context_window(...)`
8. **Edit-mode Helpers**
    1. `extract_between_markers(...)` for strict `<<BEGIN_FILE>> … <<END_FILE>>`,
    2. `lang_for(...)` (for informative context blocks)
9. **Banner**
    1. `print_mode_banner(...)` unified banner for OpenAI and Ollama frontends

**Why it matters:** this consolidation removes duplication and keeps the front-end scripts thin.

---

### 2.2 `rag/query_openai.py` (Q&A with OpenAI)

1. Uses shared utilities for config, retrieval, prompt formatting, token guard, and banner.
2. **Prompt:** `QA_SYSTEM_PROMPT` with instructions to cite chunks and stick to repository context.
3. **CLI:**
    1. `--top-k` to control context breadth,
    2. `--show-sources` to print retrieved chunk references,
    3. `--model` to switch models at runtime.
4. **Rerank:** optional (`RAG_RERANK=1`) via cross-encoder (`RAG_RERANK_MODEL`), with `RERANK_K` as the over-fetch size.

---

### 2.3 `rag/query_ollama.py` (Q&A with Ollama)

1. Parity with `query_openai.py`, but generation is sent to **Ollama**.
2. Context window and max output use Ollama-family heuristics.
3. Same retrieval strategy (structure slice + hints + vector fill).
4. Uses the shared banner, prompt formatting, and token guard.

---

### 2.4 `rag/query_openai_file.py` (single-file editor with OpenAI)

1. **Purpose:** Rewrite a single file and emit **only** the fully updated file.
2. **Strict I/O contract.**
    1. The model **must** return the complete file wrapped between markers:
        ```
        <<BEGIN_FILE>>
        # ...entire updated file...
        <<END_FILE>>
        ```
    2. The script extracts only the payload between the markers.
3. **Context construction:**
    1. Small `STRUCTURE.md` slice (reserved slots),
    2. Filename-hinted + vector-retrieved chunks **excluding** the target file (prevents parroting),
    3. The **entire current file** is injected as “source of truth.”
4. **Dynamic token budgeting:**
    1. Computes `computed_max_output = min(requested_max_output, ctx_window - input_tokens - safety)`,
    2. Prints the computed max tokens before sending the request,
    3. Applies `ensure_prompt_fits(...)` with the computed value.
5. **Safety & determinism:**
    1. `temperature=0.0`, strict markers, and clear constraints.
6. **`--rewrite`:**
    1. If supplied, writes the updated content back to the original file path (assumes version control is guarding against irrevocable loss).

---

## 3) Retrieval Strategy

1. **Filename hints (priority).**  
   If the prompt mentions files like `registry.py`, those basenames are resolved through the **manifest** collection into full paths. A small capped number of chunks per hinted file are injected first.
2. **Structure slice (reserved).**  
   A small number of chunks from `rag/.generated/STRUCTURE.md` are injected to give the model a global layout view. This prevents the editor/QA from hallucinating directories or missing module counts.
3. **Vector search fill.**  
   The remainder of the context is filled with standard vector search results from the main code chunk collection. Optional re-ranking can be applied across the union (hints + structure + raw).
4. **Exclusions (edit mode).**  
   When editing, the target file’s chunks are excluded from retrieval so the model cannot simply echo the original content. The canonical source is the explicit “Target file (current contents)” section.

---

## 4) Prompting

1. **Q&A prompts** use `QA_SYSTEM_PROMPT` (from utils) with appended instructions to:
    1. cite chunks by index and path,
    2. give minimal, correct code when needed,
    3. rely on `STRUCTURE.md` when present,
    4. admit “don’t know” if the answer isn’t in context.
2. **Edit prompts** use a strict **edit system prompt** (in the file editor script) requiring the full file between markers and forbidding commentary, headers, or code fences.
3. **Overrides:** You can override the base QA system prompt without editing code:
    1. `RAG_SYS_PROMPT="...your text..."`, or
    2. `RAG_SYS_PROMPT_FILE=/abs/path/to/prompt.txt`.

---

## 5) Token Budgeting

1. `resolve_context_window(model, map, default)` picks a conservative context limit for each model family.
2. `count_tokens(...)` uses `tiktoken` if available (fallback heuristic otherwise).
3. `ensure_prompt_fits(...)` enforces `input_tokens + max_output + safety <= context_window`.
4. The file editor computes **dynamic** `max_output` per request and prints the final value.

---

## 6) Configuration

Create a `.env` or set env vars in your shell. Sensible code-first defaults are used when possible.

### 6.1 Core Paths

| Variable             | Default                             | Purpose                                    |
| -------------------- | ----------------------------------- | ------------------------------------------ |
| `RAG_REPO_ROOT`      | project root (parent of `rag/`)     | Repository base for relative paths         |
| `RAG_INDEX_DIR`      | `.rag_index/` under `RAG_REPO_ROOT` | Chroma persistent store                    |
| `RAG_COLLECTION`     | `code_chunks`                       | Base name for collections                  |
| `RAG_STRUCTURE_PATH` | `rag/.generated/STRUCTURE.md`       | Structure file used as small context slice |

### 6.2 Embeddings & Retrieval

| Variable                   | Default                                                              | Notes                                   |
| -------------------------- | -------------------------------------------------------------------- | --------------------------------------- |
| `RAG_EMBED_BACKEND`        | `openai` (Q&A OpenAI) / `sbert` (Ollama)                             | `openai` or `sbert`                     |
| `RAG_EMBED_MODEL`          | `text-embedding-3-large` or `sentence-transformers/all-MiniLM-L6-v2` | Auto-selected by backend if not set     |
| `RAG_TOP_K`                | `6`                                                                  | Total retrieved chunks target           |
| `RAG_RERANK`               | `0`                                                                  | Set `1` to enable cross-encoder re-rank |
| `RAG_RERANK_K`             | `max(TOP_K, 20)`                                                     | Over-fetch size for re-rank             |
| `RAG_STRUCTURE_MAX_CHUNKS` | `1`                                                                  | Reserved STRUCTURE.md slices            |
| `RAG_FILE_HINT_MAX_CHUNKS` | `8`                                                                  | Per-file cap for hinted files           |

### 6.3 Generation

| Variable                   | Default                               | Notes                                        |
| -------------------------- | ------------------------------------- | -------------------------------------------- |
| `RAG_OPENAI_MODEL`         | `gpt-4o-mini`                         | OpenAI model for Q&A and editing             |
| `OPENAI_API_KEY`           | *(required)*                          | Needed for OpenAI embeddings and/or gen      |
| `RAG_OLLAMA_MODEL`         | `llama3`                              | Ollama model name                            |
| `OLLAMA_URL`               | `http://localhost:11434/api/generate` | Ollama endpoint                              |
| `RAG_SAFETY_MARGIN_TOKENS` | `1024` (OpenAI) / `512` (Ollama)      | Safety headroom                              |
| `OPENAI_MAX_OUTPUT`        | `4096` (Q&A default)                  | File editor computes final value dynamically |

---

## 7) Usage

### 7.1 Q&A (OpenAI)

```bash
python -m rag.query_openai "How is the registry initialized?" --show-sources
# Optional model override:
python -m rag.query_openai "What writes the DOT graph?" --model gpt-4o
# Adjust retrieval breadth:
python -m rag.query_openai "Where is STRUCTURE.md produced?" --top-k 10
````

### 7.2 Q&A (Ollama)

```bash
python -m rag.query_ollama "Explain emitters pipeline."
# With more context and sources:
python -m rag.query_ollama "How are chunks stored?" --top-k 10 --show-sources
```

### 7.3 Edit a Single File (OpenAI)

```bash
python -m rag.query_openai_file "src/knime2py/implemented_cli.py" \
  --edit "Add meaningful docstrings to all public functions based on the code." \
  --rewrite
```

1. The script prints a banner.
2. It logs the **computed** `max_output` before requesting completion.
3. It prints only the updated file (and rewrites in place if `--rewrite` is present).

### 7.4 Bulk docstring insertion (example)

```bash
#!/usr/bin/env bash
set -euo pipefail
TARGET_DIR="${1:-src/}"

find "$TARGET_DIR" -type f -name '*.py' ! -name '__*.py' -print0 \
| while IFS= read -r -d '' PYFILE; do
  echo "[RAG] Editing: $PYFILE"
  python -m rag.query_openai_file \
    "$PYFILE" \
    --edit "Add meaningful docstrings to all public functions based on the code." \
    --rewrite
done
```

---

## 8) Operational Notes

1. **STRUCTURE.md** must exist and be chunked into the index; otherwise the scripts still work but lose the layout hints. Keep the reserved slice **small** (e.g., 1 chunk).
2. **Index lifecycle.** If retrieval returns nothing, you likely have:

   1. a missing or stale `./.rag_index/`,
   2. embed backend/model mismatch (collection name differs),
   3. or wrong repo root. Rebuild/reindex and ensure envs match.
3. **Edit mode exclusions.** The editor deliberately excludes the target file from retrieval to avoid parroting. The source of truth for the target file is the injected “Target file (current contents)” block.
4. **Token budgeting.** If you hit context limits:

   1. lower `--top-k`, `RAG_STRUCTURE_MAX_CHUNKS`, or `RAG_FILE_HINT_MAX_CHUNKS`,
   2. reduce the size of your request text,
   3. or switch to a larger-context model.

---

## 9) Troubleshooting

1. **“RAG index directory not found … Build the index first.”**
   The Chroma DB at `./.rag_index/` is missing. Rebuild the index with your indexing script (not included here).
2. **“No context retrieved …”**
   Index is empty or collections don’t match your current `RAG_EMBED_BACKEND`/`RAG_EMBED_MODEL`. Align envs and reindex.
3. **Marker errors in edit mode.**
   The model must return the file **only** between `<<BEGIN_FILE>>` and `<<END_FILE>>`. If violated, the script fails fast. Re-run or tighten the request.
4. **OpenAI key missing.**
   Set `OPENAI_API_KEY` in `.env` or shell for OpenAI embeddings/generation. For Ollama + SBERT you can avoid OpenAI entirely.

---

## 10) Security & Safety

1. Secrets are not embedded in code. Only `OPENAI_API_KEY` is required for OpenAI paths.
2. The `--rewrite` flag writes in place. Use it on files under version control only. Review diffs.

---

## 11) Known Limitations / Next Steps

1. **Indexing pipeline** is assumed but not documented here (chunk size/overlap, filters). Add a reproducible `rag/index_repo.py` with deterministic chunking and a file manifest writer.
2. **Deduping & diversity.** Current retrieval favors hinted files and structure. Consider MMR or diversity-aware selection when `TOP_K` grows.
3. **Reranking latency.** Cross-encoder reranking adds latency; enable only when needed (`RAG_RERANK=1`).
4. **Non-Python hints.** Filename hints target `*.py` today. Extend the hint regex if you want JS/TS/MD, etc.
5. **Streaming.** Current calls are non-streaming for simplicity.

---

## 12) Minimal `.env` Example

```env
# Index + collection
RAG_REPO_ROOT=.
RAG_INDEX_DIR=.rag_index
RAG_COLLECTION=code_chunks
RAG_STRUCTURE_PATH=rag/.generated/STRUCTURE.md

# Retrieval
RAG_EMBED_BACKEND=openai        # or sbert
RAG_EMBED_MODEL=text-embedding-3-large

# Generation
RAG_OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Optional
RAG_TOP_K=6
RAG_STRUCTURE_MAX_CHUNKS=1
RAG_FILE_HINT_MAX_CHUNKS=8
RAG_RERANK=0
RAG_RERANK_K=20
RAG_SAFETY_MARGIN_TOKENS=1024
```


