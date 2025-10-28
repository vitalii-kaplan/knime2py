#!/usr/bin/env python3
"""
clean_knime_project.py

Usage:
  python clean_knime_project.py /path/to/knime/project

Behavior:
1) For each immediate subdirectory of the given path, delete everything inside it
   except the file 'settings.xml'. (Subdirectories inside those are removed entirely.)
2) In the given path itself, delete all files (including hidden) except 'workflow.knime'.
   Directories in the given path are **not** deleted.

Hidden files are processed like any other (i.e., deleted unless explicitly preserved).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


ALLOWED_NODE_FILE = "settings.xml"
ALLOWED_ROOT_FILE = "workflow.knime"


def is_fs_root(p: Path) -> bool:
    """Return True if p appears to be a filesystem root (/, C:\\, etc.)."""
    try:
        p = p.resolve()
    except Exception:
        p = p
    # On POSIX, root is "/"; on Windows, root is like "C:\\"
    return p == p.anchor and p.name == ""


def clean_node_dir(node_dir: Path) -> None:
    """
    Delete all contents of `node_dir` except the file 'settings.xml'.
    Any directories within are removed entirely.

    Args:
        node_dir (Path): The path to the node directory to clean.
    """
    if not node_dir.is_dir():
        return

    for entry in node_dir.iterdir():
        # Keep exactly 'settings.xml' (case-sensitive)
        if entry.is_file() and entry.name == ALLOWED_NODE_FILE:
            continue

        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=False)
            else:
                # includes regular files, hidden files, symlinks
                entry.unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] Failed to remove {entry}: {e}", file=sys.stderr)


def clean_root_files(root: Path) -> None:
    """
    In the root path, delete all files (including hidden) except 'workflow.knime'.
    Do not delete any directories in the root.

    Args:
        root (Path): The path to the root directory to clean.
    """
    for entry in root.iterdir():
        if entry.is_dir():
            # Do not delete directories here
            continue
        # Keep exactly 'workflow.knime' (case-sensitive)
        if entry.is_file() and entry.name == ALLOWED_ROOT_FILE:
            continue

        try:
            entry.unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] Failed to remove {entry}: {e}", file=sys.stderr)


def main() -> None:
    """
    Main function to parse command line arguments and initiate the cleaning process
    for the specified KNIME project directory.
    """
    ap = argparse.ArgumentParser(description="Clean KNIME project: keep settings.xml in node dirs and workflow.knime in root.")
    ap.add_argument("path", help="Path to the KNIME project directory")
    args = ap.parse_args()

    root = Path(args.path).expanduser()

    if not root.exists():
        print(f"[ERROR] Path does not exist: {root}", file=sys.stderr)
        sys.exit(1)
    if not root.is_dir():
        print(f"[ERROR] Path is not a directory: {root}", file=sys.stderr)
        sys.exit(1)
    if is_fs_root(root):
        print(f"[ERROR] Refusing to operate on filesystem root: {root}", file=sys.stderr)
        sys.exit(2)

    # 1) Clean each immediate subdirectory
    for entry in root.iterdir():
        if entry.is_dir():
            clean_node_dir(entry)

    # 2) Clean files in root (keep workflow.knime)
    clean_root_files(root)

    print("[OK] Cleaning complete.")


if __name__ == "__main__":
    main()
