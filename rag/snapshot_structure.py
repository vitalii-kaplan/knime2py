# rag/snapshot_structure.py
# Generate rag/.generated/STRUCTURE.md describing the repo tree.
# Uses the same ignore rules as the indexers:
#   - EXCLUDE_DIRS (hard excludes)
#   - patterns from .ragignore (gitignore-like, fnmatch on native and POSIX paths)

from __future__ import annotations

import fnmatch
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

# Optional: load .env before reading env vars (harmless if missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ---------- Config (kept in code; env can override if you really want) ----------
REPO_ROOT = Path(os.getenv("RAG_REPO_ROOT", Path(__file__).resolve().parents[1]))
OUT = REPO_ROOT / "rag" / ".generated" / "STRUCTURE.md"

EXCLUDE_DIRS = {
    "__pycache__", ".conda", ".git", ".pytest_cache", ".rag_index",
    ".venv", ".venv-pex", "venv", ".vscode", "dist", "output", "build",
}

# ---------- Ignore helpers (mirrors indexer logic) ----------
def _read_ragignore(root: Path) -> List[str]:
    """Load ignore patterns from .ragignore at repo root."""
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
    """fnmatch on both native and POSIX-style relative paths."""
    posix = rel_path.replace(os.sep, "/")
    return any(fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(posix, p) for p in patterns)

def _skip_dir_reason(rel_dir: str, patterns: Sequence[str]) -> str | None:
    """
    Return 'default' if EXCLUDE_DIRS applies,
    'ragignore' if .ragignore matches, else None.
    """
    if not rel_dir or rel_dir == ".":
        parts = set()
    else:
        parts = set(Path(rel_dir).parts)
    if parts & EXCLUDE_DIRS:
        return "default"
    if _is_ignored(rel_dir, patterns):
        return "ragignore"
    return None

# ---------- Structure writer ----------
def _walk_filtered(root: Path, patterns: Sequence[str]) -> Iterable[tuple[str, list[str], list[str]]]:
    """
    os.walk wrapper that prunes ignored dirs (EXCLUDE_DIRS + .ragignore)
    with the same top-down semantics as the indexers.
    Yields (rel_dir, dirnames, filenames) with rel_dir relative to root ('' for top).
    """
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""

        # If current directory is ignored, stop descent entirely.
        reason = _skip_dir_reason(rel_dir, patterns)
        if reason is not None:
            dirnames[:] = []
            continue

        # Otherwise, prune child directories precisely.
        kept: list[str] = []
        for d in dirnames:
            child_rel = os.path.normpath(os.path.join(rel_dir, d)) if rel_dir else d
            if d in EXCLUDE_DIRS:
                continue
            if _is_ignored(child_rel, patterns):
                continue
            kept.append(d)
        dirnames[:] = kept

        yield rel_dir, dirnames, filenames

def _header_block(ignore_patterns: Sequence[str]) -> List[str]:
    """Build a formal, informative header for STRUCTURE.md."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: List[str] = []
    lines.append("# Project structure")
    lines.append("")
    lines.append("> **Purpose**: This document is an automatically generated snapshot of the repository’s directory")
    lines.append("> and file layout. It is intended for documentation, navigation, and retrieval-augmented")
    lines.append("> generation (RAG). It lists paths only (no file contents).")
    lines.append("> ")
    lines.append("> **Scope**: The listing reflects the tree under the repository root, excluding paths filtered by")
    lines.append("> `.ragignore` patterns and built-in excludes (e.g., `.git`, virtual environments, build artifacts).")
    lines.append("> ")
    lines.append(f"> **Generated**: {ts}")
    if ignore_patterns:
        lines.append("> **Active ignore sources**:")
        lines.append("> - `.ragignore` (pattern matching on native and POSIX paths)")
        lines.append("> - Built-in excluded directories defined by the snapshot script")
    else:
        lines.append("> **Active ignore sources**:")
        lines.append("> - Built-in excluded directories defined by the snapshot script")
    lines.append("")
    return lines

def generate_structure_md() -> str:
    ignore_patterns = _read_ragignore(REPO_ROOT)

    lines: List[str] = []
    lines.extend(_header_block(ignore_patterns))

    # Walk and render
    for rel_dir, dirnames, filenames in _walk_filtered(REPO_ROOT, ignore_patterns):
        # Sort to keep output stable
        dirnames.sort()
        filenames_sorted = sorted(filenames)

        # Directory heading
        if rel_dir:
            depth = rel_dir.count(os.sep)
            lines.append(f"{'  ' * depth}- **{rel_dir}/**")

        # Files
        depth_files = (rel_dir.count(os.sep) + 1) if rel_dir else 0
        indent = "  " * depth_files
        for f in filenames_sorted:
            rel = str(Path(rel_dir, f)) if rel_dir else f
            if _is_ignored(rel, ignore_patterns):
                continue
            lines.append(f"{indent}- {f}")

    return "\n".join(lines) + "\n"

def main() -> None:
    content = generate_structure_md()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(content, encoding="utf-8")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
