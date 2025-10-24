# rag/query.py
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

# ---------- Config (env overridable) ----------
REPO_ROOT = Path(os.getenv("RAG_REPO_ROOT", Path(__file__).resolve().parents[1]))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", REPO_ROOT / ".rag_index"))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "code_chunks")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

BACKEND = os.getenv("RAG_BACKEND", "ollama").lower()  # "ollama" or "openai"
OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", "llama3")
OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
RERANK = os.getenv("RAG_RERANK", "0") == "1"
RERANK_K = int(os.getenv("RAG_RERANK_K", str(max(TOP_K, 20))))
MAX_CTX_CHARS = int(os.getenv("RAG_MAX_CTX_CHARS", "12000"))

SYS = (
    "You are a coding assistant. Answer strictly from the provided repository context. "
    "If unsure, say you don't know. Include file paths next to relevant explanations."
)


# ---------- LLM backends ----------
def llm_openai(prompt: str) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise ImportError(
            "OpenAI client not installed. Install with `pip install openai` or use BACKEND=ollama."
        ) from e
    client = OpenAI()  # requires OPENAI_API_KEY in env
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def llm_ollama(prompt: str) -> str:
    import requests

    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    data = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    r = requests.post(url, json=data, timeout=600)
    r.raise_for_status()
    js = r.json()
    return js.get("response", "")


def _choose_llm():
    if BACKEND == "openai":
        return llm_openai
    return llm_ollama


# ---------- Retrieval ----------
def _get_collection():
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"RAG index directory not found: {INDEX_DIR}. Run `python -m rag.index_repo --full` first."
        )
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found in {INDEX_DIR}. Build the index first."
        ) from e


def _retrieve_raw(query: str, n: int):
    coll = _get_collection()
    embedder = SentenceTransformer(EMBED_MODEL)
    q_emb = embedder.encode([query], normalize_embeddings=True)
    res = coll.query(query_embeddings=q_emb, n_results=n, include=["documents", "metadatas"])
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return list(zip(docs, metas))


def _maybe_rerank(query: str, passages: List[Tuple[str, dict]], top_k: int) -> List[Tuple[str, dict]]:
    if not RERANK:
        return passages[:top_k]
    try:
        # Lightweight cross-encoder works well for code/doc passages
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception:
        # Reranker not available; fall back gracefully
        return passages[:top_k]

    model_name = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoder(model_name)
    pairs = [(query, p[0]) for p in passages]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(passesages_with_meta := passages, scores), key=lambda x: float(x[1]), reverse=True)
    return [pm for (pm, _score) in ranked[:top_k]]


def retrieve(query: str, top_k: int = TOP_K) -> List[Tuple[str, dict]]:
    # Pull more for reranker if enabled
    n = max(top_k, RERANK_K if RERANK else top_k)
    raw = _retrieve_raw(query, n)
    return _maybe_rerank(query, raw, top_k)


# ---------- Prompting ----------
def make_prompt(query: str, passages: Sequence[Tuple[str, dict]], max_chars: int = MAX_CTX_CHARS) -> str:
    context_chunks: List[str] = []
    total = 0
    for i, (doc, meta) in enumerate(passages, 1):
        header = f"[{i}] {meta.get('path','?')}#chunk-{meta.get('chunk','?')}\n"
        block = header + str(doc)
        if total + len(block) > max_chars:
            break
        context_chunks.append(block)
        total += len(block)

    ctx = "\n\n".join(context_chunks)
    return (
        f"{SYS}\n\nQuestion:\n{query}\n\n"
        f"Repository context:\n{ctx}\n\n"
        "Instructions:\n"
        "- Cite chunks by [index] and file path.\n"
        "- If code is needed, provide minimal, correct snippets.\n"
        "- If the answer is not in the context, say you don't know.\n"
    )


def ask(query: str, show_sources: bool = False) -> str:
    passages = retrieve(query)
    if not passages:
        return "No context retrieved. Rebuild the index or broaden your query."
    prompt = make_prompt(query, passages)
    ans = _choose_llm()(prompt)

    if show_sources:
        srcs = "\n".join(
            f"[{i}] {m.get('path','?')}#chunk-{m.get('chunk','?')}"
            for i, (_d, m) in enumerate(passages, 1)
        )
        return f"{ans}\n\nSources:\n{srcs}"
    return ans


def main(argv: Sequence[str] | None = None) -> int:
    global TOP_K

    p = argparse.ArgumentParser(description="Query local RAG index over the repository.")
    p.add_argument("question", nargs="*", help="Your question about the codebase")
    p.add_argument("--top-k", type=int, default=TOP_K, help="Number of passages to feed to the LLM")
    p.add_argument("--show-sources", action="store_true", help="Append source chunk references to the answer")
    args = p.parse_args(argv)

    TOP_K = args.top_k

    q = " ".join(args.question).strip() or "How do we initialize the DB client?"
    try:
        print(ask(q, show_sources=args.show_sources))
        return 0
    except KeyboardInterrupt:
        print("Aborted.")
        return 130
    except Exception as e:
        # Fail loudly with a clear action
        print(f"Error: {e}\nHint: Did you run `python -m rag.index_repo --full` and set BACKEND correctly?")
        return 1


if __name__ == "__main__":
    sys.exit(main())
