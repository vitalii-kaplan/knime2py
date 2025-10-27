# rag/query_ollama.py
# Query with Ollama using shared RAG utilities and a token-budget guard.

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

# Load .env before reading env vars (harmless if missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import requests

from rag.rag_utils import (
    RAGConfig,
    load_config_from_env,
    retrieve_with_structure_and_hints,
    ensure_prompt_fits,
    resolve_context_window,
    print_mode_banner,
    build_qa_prompt,
    system_prompt,
)

# ---------------- Config ----------------
# For Ollama we default retrieval backend to SBERT unless env overrides.
CFG: RAGConfig = load_config_from_env(
    default_embed_backend="sbert",
    default_openai_model="text-embedding-3-small",
    default_sbert_model="sentence-transformers/all-MiniLM-L6-v2",
)

STRUCTURE_MAX_CHUNKS = int(os.getenv("RAG_STRUCTURE_MAX_CHUNKS", "1"))
FILE_HINT_MAX_CHUNKS = int(os.getenv("RAG_FILE_HINT_MAX_CHUNKS", "8"))

OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", "llama3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

TOP_K = int(os.getenv("RAG_TOP_K", "6"))
RERANK = os.getenv("RAG_RERANK", "0") == "1"
RERANK_K = int(os.getenv("RAG_RERANK_K", str(max(TOP_K, 20))))

# ---- Prompt size control (token-based) ----
_OLLAMA_CTX_MAP = {
    "llama3":   8_192,
    "llama3.1": 128_000,
    "llama3.2": 128_000,
    "qwen2.5":  32_000,
    "mistral":  8_192,
    "mixtral":  32_768,
    "phi3":     4_000,
    "phi4":     8_192,
}
OLLAMA_CONTEXT_WINDOW = resolve_context_window(OLLAMA_MODEL.lower(), _OLLAMA_CTX_MAP, default=8_192)
OLLAMA_MAX_OUTPUT = int(os.getenv("OLLAMA_MAX_OUTPUT", "1024"))
RAG_SAFETY_MARGIN_TOKENS = int(os.getenv("RAG_SAFETY_MARGIN_TOKENS", "512"))

# ---------------- LLM (Ollama) ----------------
def llm_ollama(prompt: str) -> str:
    ensure_prompt_fits(
        prompt,
        ctx_limit=OLLAMA_CONTEXT_WINDOW,
        max_output=OLLAMA_MAX_OUTPUT,
        safety=RAG_SAFETY_MARGIN_TOKENS,
        model_name=OLLAMA_MODEL,
        tokenizer_hint="cl100k_base",
    )
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

# ---------------- Prompting ----------------
def make_prompt(question: str, passages) -> str:
    return build_qa_prompt(system_prompt(), question, passages)

def ask(question: str, show_sources: bool = False) -> str:
    passages = retrieve_with_structure_and_hints(
        CFG, question, TOP_K, STRUCTURE_MAX_CHUNKS, FILE_HINT_MAX_CHUNKS,
        rerank=RERANK, rerank_k=RERANK_K,
    )
    if not passages:
        return "No context retrieved. Ensure STRUCTURE.md is generated and the index matches your embed backend/model."
    prompt = make_prompt(question, passages)
    ans = llm_ollama(prompt)
    if show_sources:
        srcs = "\n".join(
            f"[{i}] {m.get('path','?')}#chunk-{m.get('chunk','?')}"
            for i, (_d, m) in enumerate(passages, 1)
        )
        return f"{ans}\n\nSources:\n{srcs}"
    return ans

# ---------------- CLI ----------------
def main(argv: Sequence[str] | None = None) -> int:
    global TOP_K

    p = argparse.ArgumentParser(description="Query repo RAG (generation=Ollama).")
    p.add_argument("question", nargs="*", help="Your question about the codebase")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--show-sources", action="store_true")
    args = p.parse_args(argv)
    TOP_K = args.top_k

    # Banner after resolving args (so top_k reflects the CLI override)
    print_mode_banner(
        CFG,
        gen_label=f"ollama({OLLAMA_MODEL})",
        model=OLLAMA_MODEL,
        top_k=TOP_K,
        rerank=RERANK,
        context_window=OLLAMA_CONTEXT_WINDOW,
        max_output=OLLAMA_MAX_OUTPUT,
        safety=RAG_SAFETY_MARGIN_TOKENS,
        extras=[f"OLLAMA_URL={OLLAMA_URL}", f"struct_slots={STRUCTURE_MAX_CHUNKS}", f"file_hint_slots={FILE_HINT_MAX_CHUNKS} per file"],
        warn_openai_embed=True,   # warn if using OpenAI embeddings without key
        warn_openai_gen=False,    # generation is Ollama
    )

    q = " ".join(args.question).strip() or "How do we initialize the DB client?"
    try:
        print(ask(q, show_sources=args.show_sources))
        return 0
    except KeyboardInterrupt:
        print("Aborted.")
        return 130
    except Exception as e:
        print(
            f"Error: {e}\nHint: Regenerate STRUCTURE.md and reindex; "
            f"reduce TOP_K or chunk size; or increase the model context."
        )
        return 1

if __name__ == "__main__":
    sys.exit(main())
