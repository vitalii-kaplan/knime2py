# rag/query_openai.py
# Generate answers with OpenAI. Retrieval uses the index that matches RAG_EMBED_BACKEND+RAG_EMBED_MODEL.
# Improvements:
#   - Inject only a small number of STRUCTURE.md chunks (reserved slots).
#   - If the question mentions specific files (e.g., registry.py), pull chunks for those files first.
#   - Fill remaining slots with normal vector search results.

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

# Load .env before any getenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
REPO_ROOT = Path(os.getenv("RAG_REPO_ROOT", Path(__file__).resolve().parents[1]))
INDEX_DIR  = Path(os.getenv("RAG_INDEX_DIR", REPO_ROOT / ".rag_index"))
COLLECTION_BASE = os.getenv("RAG_COLLECTION", "code_chunks")
STRUCTURE_PATH = os.getenv("RAG_STRUCTURE_PATH", "rag/.generated/STRUCTURE.md")

# Retrieval (embedding backend + model)
EMBED_BACKEND = os.getenv("RAG_EMBED_BACKEND", "openai").lower()  # 'sbert' or 'openai'
if EMBED_BACKEND not in {"sbert", "openai"}:
    raise ValueError("RAG_EMBED_BACKEND must be 'sbert' or 'openai'.")

_default_embed_model = (
    "text-embedding-3-large" if EMBED_BACKEND == "openai" else "sentence-transformers/all-MiniLM-L6-v2"
)
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", _default_embed_model)

# Generation (fixed to OpenAI here)
OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")

TOP_K = int(os.getenv("RAG_TOP_K", "6"))
RERANK = os.getenv("RAG_RERANK", "0") == "1"
RERANK_K = int(os.getenv("RAG_RERANK_K", str(max(TOP_K, 20))))

# Reserved slots so STRUCTURE.md doesn't crowd out real content
STRUCTURE_MAX_CHUNKS = int(os.getenv("RAG_STRUCTURE_MAX_CHUNKS", "1"))
FILE_HINT_MAX_CHUNKS = int(os.getenv("RAG_FILE_HINT_MAX_CHUNKS", "8"))  # per hinted file

# --- Prompt size control (token-based) ---
MODEL_CTX = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "o4-mini": 200_000,
    "o3": 200_000,
}
OPENAI_CONTEXT_WINDOW: int = MODEL_CTX.get(OPENAI_MODEL, 128_000)
OPENAI_MAX_OUTPUT = int(os.getenv("OPENAI_MAX_OUTPUT", "4096"))
RAG_SAFETY_MARGIN_TOKENS = int(os.getenv("RAG_SAFETY_MARGIN_TOKENS", "1024"))

SYS = (
    "You are a coding assistant. Answer strictly from the provided repository context. "
    "If unsure, say you don't know. Include file paths next to relevant explanations. "
    "If STRUCTURE.md is present in context, use it to reason about file layout and counts."
)

def _slug(s: str) -> str:
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def current_collection_name() -> str:
    return f"{COLLECTION_BASE}__emb-{_slug(EMBED_BACKEND)}-{_slug(EMBED_MODEL)}"

def _manifest_collection_name() -> str:
    return f"{COLLECTION_BASE}_files__emb-{_slug(EMBED_BACKEND)}-{_slug(EMBED_MODEL)}"

def _resolve_ctx_limit(model: str) -> int:
    if model in MODEL_CTX:
        return MODEL_CTX[model]
    for k, v in MODEL_CTX.items():
        if model.startswith(k):
            return v
    return 128_000

# ---------------- Mode banner ----------------
def _print_mode_banner() -> None:
    retr = f"{EMBED_BACKEND}:{EMBED_MODEL}"
    idx  = f"{INDEX_DIR}::{current_collection_name()}"
    rr   = "on" if RERANK else "off"
    extras = []
    if EMBED_BACKEND == "openai" and not os.getenv("OPENAI_API_KEY"):
        extras.append("WARNING: OPENAI_API_KEY not set for query embeddings")
    if not os.getenv("OPENAI_API_KEY"):
        extras.append("WARNING: OPENAI_API_KEY not set for generation")
    ctx_lim = _resolve_ctx_limit(OPENAI_MODEL)
    extras.append(f"context_window={ctx_lim}")
    extras.append(f"max_output={OPENAI_MAX_OUTPUT}")
    extras.append(f"safety_margin={RAG_SAFETY_MARGIN_TOKENS}")
    extras.append(f"structure_path={STRUCTURE_PATH}")
    extras.append(f"struct_slots={STRUCTURE_MAX_CHUNKS}")
    extras.append(f"file_hint_slots={FILE_HINT_MAX_CHUNKS} per file")
    print(
        f"[RAG] gen=openai({OPENAI_MODEL}) | retrieve={retr} | top_k={TOP_K} | rerank={rr}\n"
        f"[RAG] index={idx}\n"
        f"[RAG] " + " | ".join(extras),
        flush=True,
    )

# ---------------- Token counting ----------------
def _count_tokens_text(text: str, model: str) -> int:
    try:
        import tiktoken  # type: ignore
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(math.ceil(len(text) / 3.5 * 1.10))

def _ensure_prompt_fits(prompt_text: str, model: str) -> None:
    ctx_limit = _resolve_ctx_limit(model)
    in_tokens = _count_tokens_text(prompt_text, model)
    required_total = in_tokens + OPENAI_MAX_OUTPUT + RAG_SAFETY_MARGIN_TOKENS
    if required_total > ctx_limit:
        over = required_total - ctx_limit
        raise RuntimeError(
            "Prompt too large for model context window.\n"
            f"- Model: {model}\n"
            f"- Context window: {ctx_limit} tokens\n"
            f"- Input tokens (est.): {in_tokens}\n"
            f"- Requested max output: {OPENAI_MAX_OUTPUT}\n"
            f"- Safety margin: {RAG_SAFETY_MARGIN_TOKENS}\n"
            f"=> Required: {required_total} (over by {over} tokens)\n\n"
            "Fix: reduce context (lower TOP_K, shrink chunk size/overlap, or filter), "
            "or decrease OPENAI_MAX_OUTPUT, or switch to a larger-context model."
        )

# ---------------- LLM (OpenAI) ----------------
def llm_openai(prompt: str) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise ImportError("OpenAI client not installed. `pip install openai`.") from e
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    _ensure_prompt_fits(prompt, OPENAI_MODEL)
    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=OPENAI_MAX_OUTPUT,
    )
    return resp.choices[0].message.content or ""

# ---------------- Retrieval helpers ----------------
def _client() -> chromadb.PersistentClient:
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"RAG index directory not found: {INDEX_DIR}. Build the index first.")
    return chromadb.PersistentClient(path=str(INDEX_DIR))

def _get_collection():
    return _client().get_collection(current_collection_name())

def _get_manifest():
    return _client().get_collection(_manifest_collection_name())

def encode_query(q: str) -> List[float]:
    if EMBED_BACKEND == "openai":
        from openai import OpenAI  # type: ignore
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set for retrieval.")
        return OpenAI().embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
    return SentenceTransformer(EMBED_MODEL).encode([q], normalize_embeddings=True)[0].tolist()

def _retrieve_raw(query: str, n: int) -> List[Tuple[str, dict]]:
    coll = _get_collection()
    q_emb = encode_query(query)
    res = coll.query(query_embeddings=[q_emb], n_results=n, include=["documents", "metadatas"])
    docs, metas = res["documents"][0], res["metadatas"][0]
    return list(zip(docs, metas))

def _fetch_file_chunks_by_path(path: str, max_chunks: int = 8) -> List[Tuple[str, dict]]:
    coll = _get_collection()
    try:
        res = coll.get(where={"path": path}, include=["documents", "metadatas"])
    except Exception:
        return []
    docs = (res.get("documents") or [])[:max_chunks]
    metas = (res.get("metadatas") or [])[:max_chunks]
    return list(zip(docs, metas))

def _extract_file_hints(q: str) -> List[str]:
    # pick up things like `registry.py`, nodes/registry.py, "registry.py"
    # return basenames so we can match via manifest
    hits = re.findall(r"[A-Za-z0-9_\-./]+\.py", q)
    return [os.path.basename(h) for h in hits]

def _find_paths_by_basenames(basenames: List[str]) -> List[str]:
    """Look up full paths from the file manifest by basename."""
    try:
        man = _get_manifest()
    except Exception:
        return []
    res = man.get(include=["metadatas"])
    metas = res.get("metadatas") or []
    want = {b.lower() for b in basenames}
    out: List[str] = []
    for m in metas:
        p = m.get("path")
        if not p:
            continue
        if os.path.basename(p).lower() in want:
            out.append(p)
    # dedupe, stable-ish
    return sorted(set(out), key=lambda s: (s.count("/"), s))

def _maybe_rerank(query: str, passages: List[Tuple[str, dict]], top_k: int) -> List[Tuple[str, dict]]:
    if not RERANK:
        return passages[:top_k]
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception:
        return passages[:top_k]
    model_name = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoder(model_name)
    pairs = [(query, p[0]) for p in passages]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(passages, scores), key=lambda x: float(x[1]), reverse=True)
    return [pm for (pm, _s) in ranked[:top_k]]

def retrieve(query: str, top_k: int = TOP_K) -> List[Tuple[str, dict]]:
    """
    Retrieval order (no reranker):
      1) file-hinted chunks (e.g., if query mentions 'registry.py') up to FILE_HINT_MAX_CHUNKS per file
      2) STRUCTURE.md chunks (up to STRUCTURE_MAX_CHUNKS)
      3) normal vector search results to fill remaining slots
    With reranker on, we pass everything to reranker and keep top_k.
    """
    # 1) File-hinted chunks
    file_hints = _extract_file_hints(query)
    file_chunks: List[Tuple[str, dict]] = []
    if file_hints:
        paths = _find_paths_by_basenames(file_hints)
        for p in paths:
            file_chunks.extend(_fetch_file_chunks_by_path(p, max_chunks=FILE_HINT_MAX_CHUNKS))

    # 2) STRUCTURE.md (small reserved slice)
    struct_chunks = _fetch_file_chunks_by_path(STRUCTURE_PATH, max_chunks=STRUCTURE_MAX_CHUNKS)

    # 3) Normal vector results (pull more if reranking)
    n = max(top_k, RERANK_K if RERANK else top_k)
    raw = _retrieve_raw(query, n)

    combined = file_chunks + struct_chunks + raw
    if RERANK:
        return _maybe_rerank(query, combined, top_k)

    # No reranker: hand-pick in priority order, but cap to top_k
    out: List[Tuple[str, dict]] = []
    def _take(src: List[Tuple[str, dict]]):
        nonlocal out
        for item in src:
            if len(out) >= top_k:
                break
            out.append(item)

    _take(file_chunks)
    if len(out) < top_k:
        _take(struct_chunks)
    if len(out) < top_k:
        _take(raw)
    return out

# ---------------- Prompting ----------------
def make_prompt(query: str, passages: List[Tuple[str, dict]]) -> str:
    chunks: List[str] = []
    for i, (doc, meta) in enumerate(passages, 1):
        chunks.append(f"[{i}] {meta.get('path','?')}#chunk-{meta.get('chunk','?')}\n{doc}")
    ctx = "\n\n".join(chunks)
    return (
        f"{SYS}\n\nQuestion:\n{query}\n\nRepository context:\n{ctx}\n\n"
        "Instructions:\n"
        "- Cite chunks by [index] and file path.\n"
        "- If code is needed, provide minimal, correct snippets.\n"
        "- Use STRUCTURE.md context (if present) for file layout / counts / listings.\n"
        "- If the answer is not in the context, say you don't know.\n"
    )

def ask(query: str, show_sources: bool = False) -> str:
    passages = retrieve(query)
    if not passages:
        return "No context retrieved. Ensure STRUCTURE.md is generated and the index matches your embed backend/model."
    prompt = make_prompt(query, passages)
    ans = llm_openai(prompt)
    if show_sources:
        srcs = "\n".join(
            f"[{i}] {m.get('path','?')}#chunk-{m.get('chunk','?')}"
            for i, (_d, m) in enumerate(passages, 1)
        )
        return f"{ans}\n\nSources:\n{srcs}"
    return ans

def main(argv: Sequence[str] | None = None) -> int:
    global TOP_K
    _print_mode_banner()

    p = argparse.ArgumentParser(description="Query repo RAG (generation=OpenAI).")
    p.add_argument("question", nargs="*", help="Your question about the codebase")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--show-sources", action="store_true")
    args = p.parse_args(argv)
    TOP_K = args.top_k

    q = " ".join(args.question).strip() or "How do we initialize the DB client?"
    try:
        print(ask(q, show_sources=args.show_sources)); return 0
    except KeyboardInterrupt:
        print("Aborted."); return 130
    except Exception as e:
        print(
            f"Error: {e}\nHint: Regenerate STRUCTURE.md and reindex; "
            f"reduce TOP_K or chunk size; or increase model context."
        )
        return 1

if __name__ == "__main__":
    sys.exit(main())
