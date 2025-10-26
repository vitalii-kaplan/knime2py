# rag/index_sbert.py
# Build/refresh a Chroma index using SBERT embeddings (local).
# Prints full ignore/skip report on --full.

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence
from collections import defaultdict

# Load .env early; tame tokenizers/BLAS thread spam
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import chromadb
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
REPO_ROOT = Path(os.getenv("RAG_REPO_ROOT", Path(__file__).resolve().parents[1]))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", REPO_ROOT / ".rag_index"))
COLLECTION_BASE = os.getenv("RAG_COLLECTION", "code_chunks")

EMBED_BACKEND = "sbert"
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DEFAULT_EXTS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".sh", ".html"}
INCLUDE_EXT = {
    e.strip().lower()
    for e in (os.getenv("RAG_INCLUDE_EXT") or ",".join(sorted(DEFAULT_EXTS))).split(",")
    if e.strip()
}

EXCLUDE_DIRS = {
    "__pycache__", ".conda", ".git", ".pytest_cache", ".rag_index",
    ".venv", ".venv-pex", "venv", ".vscode", "dist", "output", "build",
}

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
MAX_FILE_MB = float(os.getenv("RAG_MAX_FILE_MB", "2.0"))
SHOW_PROGRESS = os.getenv("RAG_PROGRESS", "1") != "0"

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def current_collection_name() -> str:
    return f"{COLLECTION_BASE}__emb-{_slug(EMBED_BACKEND)}-{_slug(EMBED_MODEL)}"

def _print_mode_banner() -> None:
    print(
        f"[RAG] indexing with embed_backend={EMBED_BACKEND} model={EMBED_MODEL} "
        f"→ {INDEX_DIR}::{current_collection_name()}",
        flush=True,
    )

# ---------- Skip logging ----------
@dataclass
class SkipLog:
    dirs_default: set[str] = field(default_factory=set)
    dirs_ragignore: set[str] = field(default_factory=set)
    files_ragignore: set[str] = field(default_factory=set)
    files_ext: set[str] = field(default_factory=set)
    files_large: set[str] = field(default_factory=set)
    files_empty: set[str] = field(default_factory=set)

    def print_report(self, max_file_mb: float) -> None:
        def _print_group(title: str, items: Iterable[str]):
            items = sorted({p for p in items if p and p != "."})
            print(f"{title} [{len(items)}]:")
            if items:
                for p in items:
                    print(f"  - {p}")
            else:
                print("  (none)")
            print()
        print("\n=== RAG ignore/skip report ===")
        _print_group("Ignored directories (EXCLUDE_DIRS)", self.dirs_default)
        _print_group("Ignored directories (.ragignore)", self.dirs_ragignore)
        _print_group("Ignored files (.ragignore)", self.files_ragignore)
        _print_group("Ignored files (extension not in INCLUDE_EXT)", self.files_ext)
        _print_group(f"Skipped files (size > {max_file_mb:.1f} MB)", self.files_large)
        _print_group("Skipped files (empty/unreadable)", self.files_empty)
        print("Note: Only --full rebuild prints this report.\n")

# ---------- Utilities ----------
def _read_ragignore(root: Path) -> List[str]:
    path = root / ".ragignore"
    if not path.exists():
        return []
    pats: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pats.append(line)
    return pats

def _is_ignored(rel_path: str, patterns: Sequence[str]) -> bool:
    posix = rel_path.replace(os.sep, "/")
    return any(fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(posix, p) for p in patterns)

def _skip_dir_reason(rel_dir: str, patterns: Sequence[str]) -> str | None:
    parts = set() if not rel_dir or rel_dir == "." else set(Path(rel_dir).parts)
    if parts & EXCLUDE_DIRS:
        return "default"
    if _is_ignored(rel_dir, patterns):
        return "ragignore"
    return None

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _chunk_generic(t: str, size: int, overlap: int) -> List[str]:
    out: List[str] = []
    i = 0
    step = max(1, size - overlap)
    while i < len(t):
        chunk = t[i:i+size]
        if chunk.strip():
            out.append(chunk)
        i += step
    return out

_PY_SPLIT_RE = None
def _chunk_python(text: str, size: int, overlap: int) -> List[str]:
    import re
    global _PY_SPLIT_RE
    if _PY_SPLIT_RE is None:
        _PY_SPLIT_RE = re.compile(r"(?m)^(def|class)\s+\w+\s*[\(:]")
    idxs = [m.start() for m in _PY_SPLIT_RE.finditer(text)]
    if not idxs:
        return _chunk_generic(text, size, overlap)
    sections: List[str] = []
    for i, s in enumerate(idxs):
        e = idxs[i+1] if i+1 < len(idxs) else len(text)
        sect = text[s:e]
        if sect.strip():
            sections.append(sect)
    chunks: List[str] = []
    for sect in sections:
        if len(sect) <= size:
            chunks.append(sect)
        else:
            chunks.extend(_chunk_generic(sect, size, overlap))
    return chunks or _chunk_generic(text, size, overlap)

def _file_sha1(text: str) -> str:
    h = hashlib.sha1()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def _iter_files(root: Path, include_ext: Iterable[str], ignore_patterns: Sequence[str], skiplog: SkipLog):
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""
        reason = _skip_dir_reason(rel_dir, ignore_patterns)
        if reason is not None:
            (skiplog.dirs_default if reason == "default" else skiplog.dirs_ragignore).add(rel_dir or ".")
            dirnames[:] = []
            continue
        kept = []
        for d in dirnames:
            child_rel = os.path.normpath(os.path.join(rel_dir, d)) if rel_dir else d
            if d in EXCLUDE_DIRS:
                skiplog.dirs_default.add(child_rel)
                continue
            if _is_ignored(child_rel, ignore_patterns):
                skiplog.dirs_ragignore.add(child_rel)
                continue
            kept.append(d)
        dirnames[:] = kept
        for name in filenames:
            p = Path(dirpath) / name
            rel = os.path.relpath(p, root)
            if _is_ignored(rel, ignore_patterns):
                skiplog.files_ragignore.add(rel); continue
            if p.suffix.lower() not in include_ext:
                skiplog.files_ext.add(rel); continue
            yield p

def _chunk_file(path: Path, size: int, overlap: int) -> List[str]:
    text = _read_text(path)
    if not text.strip():
        return []
    if path.suffix.lower() == ".py":
        return _chunk_python(text, size, overlap)
    return _chunk_generic(text, size, overlap)

# ---------- Embeddings (SBERT) ----------
def embed_texts(texts: List[str]) -> List[List[float]]:
    try:
        model = SentenceTransformer(EMBED_MODEL)
    except Exception as e:
        raise ImportError(
            "SentenceTransformer model unavailable. Install extras with `pip install -e .[rag]` "
            f"and ensure EMBED_MODEL='{EMBED_MODEL}' is valid."
        ) from e
    vecs = model.encode(texts, batch_size=64, show_progress_bar=SHOW_PROGRESS, normalize_embeddings=True)
    return vecs.tolist() if hasattr(vecs, "tolist") else vecs

# ---------- Indexing ----------
def build_index(full_rebuild: bool = True) -> None:
    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    coll_name = current_collection_name()

    if full_rebuild:
        try: client.delete_collection(coll_name)
        except Exception: pass
        coll = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})
    else:
        try: coll = client.get_or_create_collection(coll_name, metadata={"hnsw:space": "cosine"})
        except Exception: coll = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})

    ignore_patterns = _read_ragignore(REPO_ROOT)
    skiplog = SkipLog()

    files = [p for p in _iter_files(REPO_ROOT, INCLUDE_EXT, ignore_patterns, skiplog)]
    if not files:
        print("No files matched. Check INCLUDE_EXT, .ragignore, and directory layout.")
        if full_rebuild: skiplog.print_report(MAX_FILE_MB)
        return

    ids: List[str] = []; docs: List[str] = []; metas: List[dict] = []
    max_bytes = int(MAX_FILE_MB * 1024 * 1024)

    for path in sorted(files, key=lambda p: str(p)):
        try: st = path.stat()
        except FileNotFoundError: continue
        rel = os.path.relpath(path, REPO_ROOT)
        if st.st_size > max_bytes: skiplog.files_large.add(rel); continue
        chunks = _chunk_file(path, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks: skiplog.files_empty.add(rel); continue
        file_sha = _file_sha1(_read_text(path))
        for j, chunk in enumerate(chunks):
            cid = hashlib.sha1(f"{rel}::{file_sha}::{j}".encode()).hexdigest()[:24]
            ids.append(cid); docs.append(chunk)
            metas.append({"path": rel, "chunk": j, "ext": path.suffix.lower(),
                          "file_sha": file_sha, "mtime": int(st.st_mtime), "size_bytes": int(st.st_size)})

    if not docs:
        print("Nothing to index after filtering and chunking.")
        if full_rebuild: skiplog.print_report(MAX_FILE_MB)
        return

    emb_docs = [f"PATH: {m['path']}\n{doc}" for doc, m in zip(docs, metas)]
    embs = embed_texts(emb_docs) 
    coll.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    print(f"Indexed {len(docs)} chunks from {len({m['path'] for m in metas})} files into {INDEX_DIR}::{coll_name}.")
    if full_rebuild: skiplog.print_report(MAX_FILE_MB)

    # Build per-file manifest rows
   
    by_file = defaultdict(lambda: {"chunks": 0, "ext": "", "size": 0, "mtime": 0})
    for m in metas:
        d = by_file[m["path"]]
        d["chunks"] += 1
        d["ext"] = m["ext"]
        d["size"] = max(d["size"], m["size_bytes"])
        d["mtime"] = max(d["mtime"], m["mtime"])

    manifest_name = f"{COLLECTION_BASE}_files__emb-{_slug(EMBED_BACKEND)}-{_slug(EMBED_MODEL)}"
    man = client.get_or_create_collection(manifest_name, metadata={"hnsw:space": "cosine"})

    man_ids, man_docs, man_metas = [], [], []
    for path, info in by_file.items():
        man_ids.append(f"file::{path}")
        # keep it textual so it’s retrievable semantically too
        man_docs.append(f"PATH: {path}\nEXT: {info['ext']}\nCHUNKS: {info['chunks']}\nSIZE: {info['size']}\n")
        man_metas.append({"path": path, "ext": info["ext"], "chunks": info["chunks"], "size": info["size"]})

    man_embs = embed_texts(man_docs)  # same backend as chunks
    man.add(ids=man_ids, documents=man_docs, embeddings=man_embs, metadatas=man_metas)
    print(f"Added manifest: {len(man_ids)} files → {manifest_name}")

def main(argv: Sequence[str] | None = None) -> int:
    global EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, MAX_FILE_MB
    p = argparse.ArgumentParser(description="Build/refresh local RAG index (SBERT).")
    p.add_argument("--refresh", action="store_true", help="Keep collection; upsert new chunks.")
    p.add_argument("--full", action="store_true", help="Delete+recreate collection first.")
    p.add_argument("--model", type=str, default=EMBED_MODEL, help="Sentence-Transformers model.")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    p.add_argument("--max-file-mb", type=float, default=MAX_FILE_MB)
    args = p.parse_args(argv)

    EMBED_MODEL = args.model
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.overlap
    MAX_FILE_MB = args.max_file_mb

    _print_mode_banner()
    full = True
    if args.refresh: full = False
    if args.full: full = True

    try:
        build_index(full_rebuild=full); return 0
    except KeyboardInterrupt:
        print("Aborted."); return 130

if __name__ == "__main__":
    sys.exit(main())
