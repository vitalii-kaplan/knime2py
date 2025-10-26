# rag/query_ollama.py
# Generate answers with Ollama. Retrieval uses the index that matches RAG_EMBED_BACKEND+RAG_EMBED_MODEL.
# Improvements:
#   - Inject only a small number of STRUCTURE.md chunks (reserved slots).
#   - If the question mentions specific files (e.g., registry.py), pull chunks for those files first.
#   - Fill remaining slots with normal vector search results.
#   - Token-aware prompt guard: checks prompt size against model context and errors if it would overflow.

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

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
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", REPO_ROOT / ".rag_index"))
COLLECTION_BASE = os.getenv("RAG_COLLECTION", "code_chunks")
STRUCTURE_PATH = os.getenv("RAG_STRUCTURE_PATH", "rag/.generated/STRUCTURE.md")

# Retrieval (embedding backend + model)
EMBED_BACKEND = os.getenv("RAG_EMBED_BACKEND", "sbert").lower()  # 'sbert' or 'openai'
if EMBED_BACKEND not in {"sbert", "openai"}:
    raise ValueError("RAG_EMBED_BACKEND must be 'sbert' or 'openai'.")

_default_embed_model = (
    "sentence-transformers/all-MiniLM-L6-v2" if EMBED_BACKEND == "sbert" else "text-embedding-3-small"
)
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", _default_embed_model)

# Generation (Ollama)
OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", "llama3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

TOP_K = int(os.getenv("RAG_TOP_K", "6"))
RERANK = os.getenv("RAG_RERANK", "0") == "1"
RERANK_K = int(os.getenv("RAG_RERANK_K", str(max(TOP_K, 20))))

# Reserved slots so STRUCTURE.md doesn't crowd out real content
STRUCTURE_MAX_CHUNKS = int(os.getenv("RAG_STRUCTURE_MAX_CHUNKS", "1"))
FILE_HINT_MAX_CHUNKS = int(os.getenv("RAG_FILE_HINT_MAX_CHUNKS", "8"))  # per hinted file

# ---- Prompt size control (token-based) ----
# Approx context windows by model family (conservative defaults).
MODEL_CTX = {
    "llama3":   8_192,     # Llama 3 8k
    "llama3.1": 128_000,   # Llama 3.1 128k
    "llama3.2": 128_000,
    "qwen2.5":  32_000,
    "mistral":  8_192,
    "mixtral":  32_768,
    "phi3":     4_000,
    "phi4":     8_192,
}

def _resolve_ctx_limit(model: str) -> int:
    ml = model.lower()
    if ml in MODEL_CTX:
        return MODEL_CTX[ml]
    for k, v in MODEL_CTX.items():
        if ml.startswith(k):
            return v
    return 8_192

OLLAMA_CONTEXT_WINDOW = _resolve_ctx_limit(OLLAMA_MODEL)            # int
OLLAMA_MAX_OUTPUT = int(os.getenv("OLLAMA_MAX_OUTPUT", "1024"))     # tokens
RAG_SAFETY_MARGIN_TOKENS = int(os.getenv("RAG_SAFETY_MARGIN_TOKENS", "512"))

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

# ---------------- Mode banner ----------------
def _print_mode_banner() -> None:
    retr = f"{EMBED_BACKEND}:{EMBED_MODEL}"
    idx = f"{INDEX_DIR}::{current_collection_name()}"
    rr = "on" if RERANK else "off"
    extras = [
        f"OLLAMA_URL={OLLAMA_URL}",
        f"context_window={OLLAMA_CONTEXT_WINDOW}",
        f"max_output={OLLAMA_MAX_OUTPUT}",
        f"safety_margin={RAG_SAFETY_MARGIN_TOKENS}",
        f"structure_path={STRUCTURE_PATH}",
        f"struct_slots={STRUCTURE_MAX_CHUNKS}",
        f"file_hint_slots={FILE_HINT_MAX_CHUNKS} per file",
    ]
    if EMBED_BACKEND == "openai" and not os.getenv("OPENAI_API_KEY"):
        extras.append("WARNING: OPENAI_API_KEY not set for query embeddings")
    print(
        f"[RAG] gen=ollama({OLLAMA_MODEL}) | retrieve={retr} | top_k={TOP_K} | rerank={rr}\n"
        f"[RAG] index={idx}\n"
        f"[RAG] " + " | ".join(extras),
        flush=True,
    )

# ---------------- Token counting ----------------
def _count_tokens_text(text: str) -> int:
    """
    Approximate token count. If tiktoken is available, use cl100k_base as a proxy;
    otherwise fall back to a conservative char->token heuristic.
    """
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(math.ceil(len(text) / 3.5 * 1.10))

def _ensure_prompt_fits(prompt_text: str, model: str) -> None:
    ctx_limit = OLLAMA_CONTEXT_WINDOW
    in_tokens = _count_tokens_text(prompt_text)
    required_total = in_tokens + OLLAMA_MAX_OUTPUT + RAG_SAFETY_MARGIN_TOKENS
    if required_total > ctx_limit:
        over = required_total - ctx_limit
        raise RuntimeError(
            "Prompt too large for model context window.\n"
            f"- Model: {model}\n"
            f"- Context window: {ctx_limit} tokens\n"
            f"- Input tokens (est.): {in_tokens}\n"
            f"- Requested max output: {OLLAMA_MAX_OUTPUT}\n"
            f"- Safety margin: {RAG_SAFETY_MARGIN_TOKENS}\n"
            f"=> Required: {required_total} (over by {over} tokens)\n\n"
            "Fix: reduce context (lower TOP_K, shrink chunk size/overlap, or filter), "
            "or decrease OLLAMA_MAX_OUTPUT, or switch to a larger-context model."
        )

# ---------------- LLM (Ollama) ----------------
def llm_ollama(prompt: str) -> str:
    import requests
    _ensure_prompt_fits(prompt, OLLAMA_MODEL)
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": OLLAMA_MAX_OUTPUT,
            "num_ctx": OLLAMA_CONTEXT_WINDOW,
        },
    }
    r = requests.post(OLLAMA_URL, json=data, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "")

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
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ImportError("OpenAI client not installed. `pip install openai`.") from e
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set for retrieval.")
        return OpenAI().embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
    # SBERT
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
    # pick up things like registry.py, nodes/registry.py, etc.; return basenames
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
        if p and os.path.basename(p).lower() in want:
            out.append(p)
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

    # No reranker: take in priority order up to top_k
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
    ans = llm_ollama(prompt)
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

    p = argparse.ArgumentParser(description="Query repo RAG (generation=Ollama).")
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
            f"reduce TOP_K or chunk size; or increase the model context."
        )
        return 1

if __name__ == "__main__":
    sys.exit(main())
