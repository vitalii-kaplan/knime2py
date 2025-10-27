# rag/query_openai.py
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence, Tuple

# Load .env before any getenv
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from .rag_utils import (
    RAGConfig,
    load_config_from_env,
    retrieve_with_structure_and_hints,
    build_qa_prompt,
    ensure_prompt_fits,
    resolve_context_window,
    print_mode_banner,
    system_prompt,   # <-- NEW
)

# ---------------- Generation config (OpenAI) ----------------
OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")

# Model → conservative context window (tokens)
_OPENAI_CTX_MAP = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "o4-mini": 200_000,
    "o3": 200_000,
}
OPENAI_CONTEXT_WINDOW = resolve_context_window(OPENAI_MODEL, _OPENAI_CTX_MAP, default=128_000)
OPENAI_MAX_OUTPUT = int(os.getenv("OPENAI_MAX_OUTPUT", "4096"))
RAG_SAFETY_MARGIN_TOKENS = int(os.getenv("RAG_SAFETY_MARGIN_TOKENS", "1024"))

# ---------------- Retrieval knobs ----------------
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
RERANK = os.getenv("RAG_RERANK", "0") == "1"
RERANK_K = int(os.getenv("RAG_RERANK_K", str(max(TOP_K, 20))))

# Reserved slots so STRUCTURE.md doesn't crowd out real content
STRUCTURE_MAX_CHUNKS = int(os.getenv("RAG_STRUCTURE_MAX_CHUNKS", "1"))
FILE_HINT_MAX_CHUNKS = int(os.getenv("RAG_FILE_HINT_MAX_CHUNKS", "8"))  # per hinted file

# ---------------- OpenAI call ----------------
def _llm_openai(prompt: str) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise ImportError("OpenAI client not installed. `pip install openai`.") from e
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    ensure_prompt_fits(
        prompt,
        ctx_limit=OPENAI_CONTEXT_WINDOW,
        max_output=OPENAI_MAX_OUTPUT,
        safety=RAG_SAFETY_MARGIN_TOKENS,
        model_name=OPENAI_MODEL,
        tokenizer_hint="cl100k_base",
    )

    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=OPENAI_MAX_OUTPUT,
    )
    return resp.choices[0].message.content or ""

# ---------------- Prompting ----------------
def make_prompt(query: str, passages: List[Tuple[str, dict]]) -> str:
    return build_qa_prompt(system_prompt(), query, passages)

# ---------------- Retrieval + Ask ----------------
def retrieve(cfg: RAGConfig, query: str, top_k: int = TOP_K) -> List[Tuple[str, dict]]:
    return retrieve_with_structure_and_hints(
        cfg,
        query=query,
        top_k=top_k,
        struct_max=STRUCTURE_MAX_CHUNKS,
        file_hint_max=FILE_HINT_MAX_CHUNKS,
        rerank=RERANK,
        rerank_k=RERANK_K,
    )

def ask(cfg: RAGConfig, query: str, show_sources: bool = False) -> str:
    passages = retrieve(cfg, query)
    if not passages:
        return "No context retrieved. Ensure STRUCTURE.md is generated and the index matches your embed backend/model."
    prompt = make_prompt(query, passages)
    ans = _llm_openai(prompt)
    if show_sources:
        srcs = "\n".join(
            f"[{i}] {m.get('path','?')}#chunk-{m.get('chunk','?')}"
            for i, (_d, m) in enumerate(passages, 1)
        )
        return f"{ans}\n\nSources:\n{srcs}"
    return ans

# ---------------- CLI ----------------
def main(argv: Sequence[str] | None = None) -> int:
    global TOP_K, OPENAI_MODEL, OPENAI_CONTEXT_WINDOW

    cfg = load_config_from_env(default_embed_backend=os.getenv("RAG_EMBED_BACKEND", "openai"))

    p = argparse.ArgumentParser(description="Query repo RAG (generation=OpenAI).")
    p.add_argument("question", nargs="*", help="Your question about the codebase")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--show-sources", action="store_true")
    p.add_argument("--model", type=str, default=OPENAI_MODEL, help="OpenAI chat model to use.")
    args = p.parse_args(argv)

    # Allow model override at runtime
    OPENAI_MODEL = args.model
    OPENAI_CONTEXT_WINDOW = resolve_context_window(OPENAI_MODEL, _OPENAI_CTX_MAP, default=128_000)
    TOP_K = args.top_k

    print_mode_banner(
        cfg,
        gen_label=f"openai({OPENAI_MODEL})",
        model=OPENAI_MODEL,
        top_k=TOP_K,
        rerank=RERANK,
        context_window=OPENAI_CONTEXT_WINDOW,
        max_output=OPENAI_MAX_OUTPUT,
        safety=RAG_SAFETY_MARGIN_TOKENS,
        extras=[f"struct_slots={STRUCTURE_MAX_CHUNKS}", f"file_hint_slots={FILE_HINT_MAX_CHUNKS} per file"],
        warn_openai_embed=True,
        warn_openai_gen=True,
    )

    q = " ".join(args.question).strip() or "How do we initialize the DB client?"
    try:
        print(ask(cfg, q, show_sources=args.show_sources))
        return 0
    except KeyboardInterrupt:
        print("Aborted.")
        return 130
    except Exception as e:
        print(
            f"Error: {e}\nHint: Regenerate STRUCTURE.md and reindex; "
            f"reduce TOP_K or chunk size; or increase model context."
        )
        return 1

if __name__ == "__main__":
    sys.exit(main())
