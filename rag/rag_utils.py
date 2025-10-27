# rag/rag_utils.py
# Shared utilities for RAG query/edit scripts (Chroma access, retrieval helpers, token tools, banners, prompts).

from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional

import chromadb

# SBERT is optional (only needed when EMBED_BACKEND='sbert')
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore


# ---------------- Config ----------------

@dataclass(frozen=True)
class RAGConfig:
    repo_root: Path
    index_dir: Path
    collection_base: str
    structure_path: str
    embed_backend: str            # 'sbert' or 'openai'
    embed_model: str


def load_config_from_env(
    *,
    default_embed_backend: str = "openai",
    default_openai_model: str = "text-embedding-3-large",
    default_sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> RAGConfig:
    """
    Code-first defaults; env can override. Only secrets (like API keys) must be in env.
    """
    repo_root = Path(os.getenv("RAG_REPO_ROOT", Path(__file__).resolve().parents[1]))
    index_dir = Path(os.getenv("RAG_INDEX_DIR", repo_root / ".rag_index"))
    collection_base = os.getenv("RAG_COLLECTION", "code_chunks")
    structure_path = os.getenv("RAG_STRUCTURE_PATH", "rag/.generated/STRUCTURE.md")

    embed_backend = os.getenv("RAG_EMBED_BACKEND", default_embed_backend).lower()
    if embed_backend not in {"sbert", "openai"}:
        raise ValueError("RAG_EMBED_BACKEND must be 'sbert' or 'openai'.")

    default_model = default_openai_model if embed_backend == "openai" else default_sbert_model
    embed_model = os.getenv("RAG_EMBED_MODEL", default_model)

    return RAGConfig(
        repo_root=repo_root,
        index_dir=index_dir,
        collection_base=collection_base,
        structure_path=structure_path,
        embed_backend=embed_backend,
        embed_model=embed_model,
    )


# ---------------- Naming helpers ----------------

def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def current_collection_name(cfg: RAGConfig) -> str:
    return f"{cfg.collection_base}__emb-{slug(cfg.embed_backend)}-{slug(cfg.embed_model)}"


def manifest_collection_name(cfg: RAGConfig) -> str:
    return f"{cfg.collection_base}_files__emb-{slug(cfg.embed_backend)}-{slug(cfg.embed_model)}"


# ---------------- Chroma helpers ----------------

def get_client(cfg: RAGConfig) -> chromadb.PersistentClient:
    if not cfg.index_dir.exists():
        raise FileNotFoundError(f"RAG index directory not found: {cfg.index_dir}. Build the index first.")
    return chromadb.PersistentClient(path=str(cfg.index_dir))


def get_collection(cfg: RAGConfig):
    return get_client(cfg).get_collection(current_collection_name(cfg))


def get_manifest(cfg: RAGConfig):
    return get_client(cfg).get_collection(manifest_collection_name(cfg))


# ---------------- Embeddings ----------------

def encode_query(cfg: RAGConfig, text: str) -> List[float]:
    if cfg.embed_backend == "openai":
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ImportError("OpenAI client not installed. `pip install openai`.") from e
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set for retrieval.")
        return OpenAI().embeddings.create(model=cfg.embed_model, input=[text]).data[0].embedding

    # SBERT branch
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not installed. `pip install sentence-transformers`.")
    return SentenceTransformer(cfg.embed_model).encode([text], normalize_embeddings=True)[0].tolist()


# ---------------- Retrieval primitives ----------------

def retrieve_raw(cfg: RAGConfig, query: str, n: int) -> List[Tuple[str, dict]]:
    coll = get_collection(cfg)
    q_emb = encode_query(cfg, query)
    res = coll.query(query_embeddings=[q_emb], n_results=n, include=["documents", "metadatas"])
    docs, metas = res["documents"][0], res["metadatas"][0]
    return list(zip(docs, metas))


def fetch_file_chunks_by_path(cfg: RAGConfig, path: str, max_chunks: int = 8) -> List[Tuple[str, dict]]:
    coll = get_collection(cfg)
    try:
        res = coll.get(where={"path": path}, include=["documents", "metadatas"])
    except Exception:
        return []
    docs = (res.get("documents") or [])[:max_chunks]
    metas = (res.get("metadatas") or [])[:max_chunks]
    return list(zip(docs, metas))


def extract_file_hints(text: str) -> List[str]:
    # capture things like src/foo/bar.py or "bar.py"
    hits = re.findall(r"[A-Za-z0-9_\-./]+\.py", text)
    return [os.path.basename(h) for h in hits]


def find_paths_by_basenames(cfg: RAGConfig, basenames: List[str]) -> List[str]:
    try:
        man = get_manifest(cfg)
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
    # de-dupe, prefer shallower paths first (stable-ish)
    return sorted(set(out), key=lambda s: (s.count("/"), s))


def structure_chunks(cfg: RAGConfig, max_chunks: int) -> List[Tuple[str, dict]]:
    if max_chunks <= 0:
        return []
    return fetch_file_chunks_by_path(cfg, cfg.structure_path, max_chunks=max_chunks)


def _maybe_rerank(query: str, passages: List[Tuple[str, dict]], top_k: int) -> List[Tuple[str, dict]]:
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


def retrieve_with_structure_and_hints(
    cfg: RAGConfig,
    query: str,
    top_k: int,
    struct_max: int,
    file_hint_max: int,
    *,
    rerank: bool = False,
    rerank_k: int = 20,
    exclude_paths: Optional[Sequence[str]] = None,
) -> List[Tuple[str, dict]]:
    """
    Retrieval order (no reranker):
      1) file-hinted chunks up to file_hint_max per file
      2) STRUCTURE.md chunks (up to struct_max)
      3) vector results to fill remaining slots
    With reranker on: merge all and keep top_k.
    """
    exclude_norm = {os.path.normpath(p) for p in (exclude_paths or [])}

    # 1) file hints
    file_chunks: List[Tuple[str, dict]] = []
    hints = extract_file_hints(query)
    if hints:
        for p in find_paths_by_basenames(cfg, hints):
            if os.path.normpath(p) in exclude_norm:
                continue
            file_chunks.extend(fetch_file_chunks_by_path(cfg, p, max_chunks=file_hint_max))

    # 2) STRUCTURE.md
    struct = structure_chunks(cfg, struct_max)

    # 3) base vector results (over-fetch for filtering and/or rerank)
    n = max(top_k, rerank_k if rerank else top_k)
    raw = retrieve_raw(cfg, query, n)
    raw_filtered = []
    for doc, meta in raw:
        p = meta.get("path", "")
        if os.path.normpath(p) in exclude_norm:
            continue
        raw_filtered.append((doc, meta))

    combined = file_chunks + struct + raw_filtered
    if rerank:
        return _maybe_rerank(query, combined, top_k)

    # priority pick
    out: List[Tuple[str, dict]] = []
    def _take(src: List[Tuple[str, dict]]):
        nonlocal out
        for item in src:
            if len(out) >= top_k:
                break
            out.append(item)

    _take(file_chunks)
    if len(out) < top_k:
        _take(struct)
    if len(out) < top_k:
        _take(raw_filtered)
    return out


# ---------------- Formatting helpers ----------------

def format_chunks(passages: List[Tuple[str, dict]]) -> str:
    lines: List[str] = []
    for i, (doc, meta) in enumerate(passages, 1):
        lines.append(f"[{i}] {meta.get('path','?')}#chunk-{meta.get('chunk','?')}\n{doc}")
    return "\n\n".join(lines)


def build_qa_prompt(system_prompt: str, question: str, passages: List[Tuple[str, dict]], extra_instructions: str = "") -> str:
    ctx = format_chunks(passages)
    instr = (
        "- Cite chunks by [index] and file path.\n"
        "- If code is needed, provide minimal, correct snippets.\n"
        "- Use STRUCTURE.md context (if present) for file layout / counts / listings.\n"
        "- If the answer is not in the context, say you don't know.\n"
    )
    if extra_instructions:
        instr = extra_instructions.rstrip() + "\n" + instr
    return (
        f"{system_prompt}\n\n"
        f"Question:\n{question}\n\n"
        f"Repository context:\n{ctx}\n\n"
        f"Instructions:\n{instr}"
    )


# ---------------- System prompt (shared) ----------------

# Default, code-first system prompt (no env required).
QA_SYSTEM_PROMPT = (
    "You are a coding assistant. Answer strictly from the provided repository context. "
    "If unsure, say you don't know. Include file paths next to relevant explanations. "
    "If STRUCTURE.md is present in context, use it to reason about file layout and counts."
)

def system_prompt(extra: Optional[str] = None) -> str:
    """
    Returns the system prompt. Code default is QA_SYSTEM_PROMPT.
    Optional overrides (ignored unless provided):
      - RAG_SYS_PROMPT (env): literal string
      - RAG_SYS_PROMPT_FILE (env): path to a file containing the prompt text
    'extra' is appended on a new line if provided.
    """
    base = os.getenv("RAG_SYS_PROMPT")
    if not base:
        p = os.getenv("RAG_SYS_PROMPT_FILE")
        if p and os.path.exists(p):
            try:
                base = Path(p).read_text(encoding="utf-8")
            except Exception:
                base = None
    if not base:
        base = QA_SYSTEM_PROMPT
    if extra:
        if not base.endswith("\n"):
            base += "\n"
        base += extra.strip() + "\n"
    return base


# ---------------- Token tools ----------------

def count_tokens(text: str, tokenizer_hint: str = "cl100k_base") -> int:
    """
    Token estimate. If tiktoken is present use `tokenizer_hint` (good default: cl100k_base),
    otherwise conservative char->token heuristic.
    """
    try:
        import tiktoken  # type: ignore
        try:
            enc = tiktoken.get_encoding(tokenizer_hint)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(math.ceil(len(text) / 3.5 * 1.10))


def ensure_prompt_fits(
    prompt_text: str,
    *,
    ctx_limit: int,
    max_output: int,
    safety: int,
    model_name: str,
    tokenizer_hint: str = "cl100k_base",
) -> None:
    in_tokens = count_tokens(prompt_text, tokenizer_hint=tokenizer_hint)
    required_total = in_tokens + max_output + safety
    if required_total > ctx_limit:
        over = required_total - ctx_limit
        raise RuntimeError(
            "Prompt too large for model context window.\n"
            f"- Model: {model_name}\n"
            f"- Context window: {ctx_limit} tokens\n"
            f"- Input tokens (est.): {in_tokens}\n"
            f"- Requested max output: {max_output}\n"
            f"- Safety margin: {safety}\n"
            f"=> Required: {required_total} (over by {over} tokens)\n\n"
            "Fix: reduce context (lower top_k or structure/file-hint slots), split your request, "
            "or switch to a larger-context model."
        )


def resolve_context_window(model_name: str, mapping: dict[str, int], default: int = 128_000) -> int:
    """Generic resolver: exact match, then prefix match, else default."""
    if model_name in mapping:
        return mapping[model_name]
    for k, v in mapping.items():
        if model_name.startswith(k):
            return v
    return default


# ---------------- Edit helpers ----------------

_BEGIN = re.compile(r"<<BEGIN_FILE>>\s*")
_END = re.compile(r"<<END_FILE>>\s*")

def extract_between_markers(text: str, start_pat: str = r"<<BEGIN_FILE>>\s*", end_pat: str = r"<<END_FILE>>\s*") -> Optional[str]:
    begin = re.compile(start_pat).search(text)
    end = re.compile(end_pat).search(text)
    if not begin or not end or end.start() <= begin.end():
        return None
    return text[begin.end():end.start()]


_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".sh": "bash",
    ".html": "html",
    ".css": "css",
    ".txt": "",
}
def lang_for(path: Path) -> str:
    return _LANG_MAP.get(path.suffix.lower(), "")


# ---------------- Banner ----------------

def print_mode_banner(
    cfg: RAGConfig,
    *,
    gen_label: str,            # e.g., "openai(gpt-4o-mini)" or "ollama(llama3)"
    model: str,
    top_k: int,
    rerank: bool,
    context_window: int,
    max_output: int,
    safety: int,
    extras: Sequence[str] | None = None,
    warn_openai_embed: bool = False,
    warn_openai_gen: bool = False,
) -> None:
    """Reusable banner printer for both OpenAI and Ollama frontends."""
    idx = f"{cfg.index_dir}::{current_collection_name(cfg)}"
    lines: List[str] = []
    if warn_openai_embed and cfg.embed_backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        lines.append("WARNING: OPENAI_API_KEY not set for query embeddings")
    if warn_openai_gen and not os.getenv("OPENAI_API_KEY"):
        lines.append("WARNING: OPENAI_API_KEY not set for generation")
    lines.extend(
        [
            f"context_window={context_window}",
            f"max_output={max_output}",
            f"safety_margin={safety}",
            f"structure_path={cfg.structure_path}",
        ]
    )
    if extras:
        lines.extend(extras)

    retr = f"{cfg.embed_backend}:{cfg.embed_model}"
    rr = "on" if rerank else "off"
    print(
        f"[RAG] gen={gen_label} | retrieve={retr} | top_k={top_k} | rerank={rr}\n"
        f"[RAG] index={idx}\n"
        f"[RAG] " + " | ".join(lines),
        flush=True,
    )
