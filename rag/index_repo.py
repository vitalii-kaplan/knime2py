# rag/index_repo.py
# On --full rebuild, this script will print ALL ignored directories and files
# (from default excludes and .ragignore), plus other skipped categories.

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import chromadb
from sentence_transformers import SentenceTransformer


# ---------- Defaults (overridable via env/CLI) ----------
REPO_ROOT = Path(os.getenv("RAG_REPO_ROOT", Path(__file__).resolve().parents[1]))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", REPO_ROOT / ".rag_index"))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "code_chunks")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Extensions to index (comma-separated env overrides)
DEFAULT_EXTS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".sh"}
INCLUDE_EXT = {
    e.strip().lower()
    for e in (os.getenv("RAG_INCLUDE_EXT") or ",".join(sorted(DEFAULT_EXTS))).split(",")
    if e.strip()
}

# Directories always excluded
EXCLUDE_DIRS = {
    "__pycache__",
    ".conda",
    ".git",
    ".pytest_cache",
    ".rag_index",
    ".venv",
    ".venv-pex",
    "venv",
    ".vscode",
    "dist",
    "output",
    "build",
}

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "400"))        # characters
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))   # characters
MAX_FILE_MB = float(os.getenv("RAG_MAX_FILE_MB", "2.0"))    # hard cap per file
SHOW_PROGRESS = os.getenv("RAG_PROGRESS", "1") != "0"


# ---------- Skip logging ----------
@dataclass
class SkipLog:
    dirs_default: set[str] = field(default_factory=set)     # EXCLUDE_DIRS
    dirs_ragignore: set[str] = field(default_factory=set)   # matched by .ragignore
    files_ragignore: set[str] = field(default_factory=set)  # matched by .ragignore
    files_ext: set[str] = field(default_factory=set)        # not in INCLUDE_EXT
    files_large: set[str] = field(default_factory=set)      # > MAX_FILE_MB
    files_empty: set[str] = field(default_factory=set)      # empty / unreadable

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
        _print_group("Ignored directories (default excludes / EXCLUDE_DIRS)", self.dirs_default)
        _print_group("Ignored directories (.ragignore)", self.dirs_ragignore)
        _print_group("Ignored files (.ragignore)", self.files_ragignore)
        _print_group("Ignored files (extension not in INCLUDE_EXT)", self.files_ext)
        _print_group(f"Skipped files (size > {max_file_mb:.1f} MB)", self.files_large)
        _print_group("Skipped files (empty/unreadable)", self.files_empty)
        print("Note: Only --full rebuild prints this report.\n")


# ---------- Utilities ----------
def _read_ragignore(root: Path) -> List[str]:
    """Load ignore patterns from .ragignore at repo root (gitignore-like, fnmatch semantics)."""
    path = root / ".ragignore"
    if not path.exists():
        return []
    patterns: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _is_ignored(rel_path: str, patterns: Sequence[str]) -> bool:
    # fnmatch on both the raw rel path and posix-normalized variant
    posix = rel_path.replace(os.sep, "/")
    return any(fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(posix, p) for p in patterns)


def _skip_dir_reason(rel_dir: str, patterns: Sequence[str]) -> str | None:
    """Return 'default' if EXCLUDE_DIRS applies, 'ragignore' if .ragignore matches, else None."""
    if not rel_dir or rel_dir == ".":
        parts = set()
    else:
        parts = set(Path(rel_dir).parts)
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
    n = len(t)
    step = max(1, size - overlap)
    while i < n:
        chunk = t[i : i + size]
        if chunk.strip():
            out.append(chunk)
        i += step
    return out


_PY_SPLIT_RE = None  # compiled lazily to avoid import at module import


def _chunk_python(text: str, size: int, overlap: int) -> List[str]:
    """Rough, fast split by top-level def/class boundaries, then sub-chunk."""
    import re

    global _PY_SPLIT_RE
    if _PY_SPLIT_RE is None:
        # Top-level 'def' or 'class' (no indentation). Multiline mode.
        _PY_SPLIT_RE = re.compile(r"(?m)^(def|class)\s+\w+\s*[\(:]")
    idxs = [m.start() for m in _PY_SPLIT_RE.finditer(text)]
    if not idxs:
        return _chunk_generic(text, size, overlap)

    # Build sections between indices
    sections: List[str] = []
    for i, s in enumerate(idxs):
        e = idxs[i + 1] if i + 1 < len(idxs) else len(text)
        sect = text[s:e]
        if sect.strip():
            sections.append(sect)

    # Sub-chunk long sections
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


def _iter_files(
    root: Path,
    include_ext: Iterable[str],
    ignore_patterns: Sequence[str],
    skiplog: SkipLog,
) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""

        # If the current directory itself is ignored, record and prune all descent.
        reason = _skip_dir_reason(rel_dir, ignore_patterns)
        if reason is not None:
            tgt = skiplog.dirs_default if reason == "default" else skiplog.dirs_ragignore
            tgt.add(rel_dir or ".")
            dirnames[:] = []
            continue

        # Otherwise, prune child directories and log them precisely.
        kept: list[str] = []
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
                skiplog.files_ragignore.add(rel)
                continue
            if p.suffix.lower() not in include_ext:
                skiplog.files_ext.add(rel)
                continue
            yield p


def _chunk_file(path: Path, size: int, overlap: int) -> List[str]:
    text = _read_text(path)
    if not text.strip():
        return []
    if path.suffix.lower() == ".py":
        return _chunk_python(text, size, overlap)
    return _chunk_generic(text, size, overlap)


# ---------- Indexing ----------
def build_index(full_rebuild: bool = True) -> None:
    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Helpful error if deps missing (when users didn't install extras)
    try:
        model = SentenceTransformer(EMBED_MODEL)
    except Exception as e:
        raise ImportError(
            "RAG embedding model is unavailable. Install extras with `pip install -e .[rag]` "
            f"and ensure the model '{EMBED_MODEL}' is valid. You can override via RAG_EMBED_MODEL."
        ) from e

    client = chromadb.PersistentClient(path=str(INDEX_DIR))

    if full_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        coll = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    else:
        try:
            coll = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        except Exception:
            coll = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ignore_patterns = _read_ragignore(REPO_ROOT)
    skiplog = SkipLog()

    files: List[Path] = [p for p in _iter_files(REPO_ROOT, INCLUDE_EXT, ignore_patterns, skiplog)]
    if not files:
        print("No files matched. Check INCLUDE_EXT, .ragignore, and directory layout.")
        if full_rebuild:
            skiplog.print_report(MAX_FILE_MB)
        return

    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []

    max_bytes = int(MAX_FILE_MB * 1024 * 1024)

    for path in sorted(files, key=lambda p: str(p)):
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        rel = os.path.relpath(path, REPO_ROOT)

        if st.st_size > max_bytes:
            skiplog.files_large.add(rel)
            continue

        chunks = _chunk_file(path, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            skiplog.files_empty.add(rel)
            continue

        # Per-file sha for provenance and de-dup reference
        file_sha = _file_sha1(_read_text(path))
        for j, chunk in enumerate(chunks):
            cid = hashlib.sha1(f"{rel}::{file_sha}::{j}".encode()).hexdigest()[:24]
            ids.append(cid)
            docs.append(chunk)
            metas.append(
                {
                    "path": rel,
                    "chunk": j,
                    "ext": path.suffix.lower(),
                    "file_sha": file_sha,
                    "mtime": int(st.st_mtime),
                    "size_bytes": int(st.st_size),
                }
            )

    if not docs:
        print("Nothing to index after filtering and chunking.")
        if full_rebuild:
            skiplog.print_report(MAX_FILE_MB)
        return

    embs = model.encode(
        docs,
        batch_size=64,
        show_progress_bar=SHOW_PROGRESS,
        normalize_embeddings=True,
    )
    coll.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    uniq_files = len({m["path"] for m in metas})
    print(
        f"Indexed {len(docs)} chunks from {uniq_files} files into {INDEX_DIR}."
    )

    # On full rebuild, print a full ignore/skip report.
    if full_rebuild:
        skiplog.print_report(MAX_FILE_MB)


def main(argv: Sequence[str] | None = None) -> int:
    # must come first because we read these names below (defaults) and assign later
    global EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, MAX_FILE_MB

    p = argparse.ArgumentParser(description="Build/refresh local RAG index for the repo.")
    p.add_argument("--refresh", action="store_true", help="Refresh (keep collection) instead of full rebuild.")
    p.add_argument("--full", action="store_true", help="Force full rebuild (delete + recreate collection).")
    p.add_argument("--model", type=str, default=EMBED_MODEL, help="Sentence-Transformers embed model name.")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    p.add_argument("--max-file-mb", type=float, default=MAX_FILE_MB)
    args = p.parse_args(argv)

    EMBED_MODEL = args.model
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.overlap
    MAX_FILE_MB = args.max_file_mb

    full = True
    if args.refresh:
        full = False
    if args.full:
        full = True

    try:
        build_index(full_rebuild=full)
        return 0
    except KeyboardInterrupt:
        print("Aborted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
