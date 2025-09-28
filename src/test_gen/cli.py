#!/usr/bin/env python3
"""
knime2py test generator helper — cleanup step

Usage:
  # Clean tests/data/<NAME>
  python -m test_gen.cli <NAME>

  # Or clean an explicit path
  python -m test_gen.cli --path /absolute/or/relative/path/to/knime/project

Options:
  --data-dir PATH   Override the default tests/data directory (defaults to <repo>/tests/data).
  --dry-run         Show what would be deleted, but do not delete.
  -v, --verbose     Print details.

Behavior:
- In each immediate subdirectory (node dir): delete everything except 'settings.xml'.
  Hidden files and directories are removed as well.
- In the project root: delete all files (including hidden) except 'workflow.knime',
  and delete hidden directories. Non-hidden directories (i.e., node directories) are kept.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ALLOWED_NODE_FILE = "settings.xml"
ALLOWED_ROOT_FILE = "workflow.knime"


# --------------------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------------------
def repo_root_from_this_file() -> Path:
    """
    Resolve the repository root, assuming this file is at: <repo>/src/test_gen/cli.py
    """
    return Path(__file__).resolve().parents[2]


def default_tests_data_dir() -> Path:
    return repo_root_from_this_file() / "tests" / "data"


def is_fs_root(p: Path) -> bool:
    """Return True if p appears to be a filesystem root (/, C:\\, etc.)."""
    try:
        p = p.resolve()
    except Exception:
        p = p
    return p == p.anchor and p.name == ""


def resolve_project_dir(project: str | None, path: str | None, data_dir: Path) -> Path:
    """
    Resolve the KNIME project directory to clean.
    - If --path is given, use it.
    - Else interpret the positional 'project' as a name under tests/data/<project>.
    """
    if path:
        p = Path(path).expanduser()
    else:
        if not project:
            raise SystemExit("ERROR: must provide either <NAME> or --path.")
        p = (data_dir / project).expanduser()
    return p.resolve()


# --------------------------------------------------------------------------------------
# Hidden detection
# --------------------------------------------------------------------------------------
def is_hidden_path(p: Path) -> bool:
    """
    Heuristic to detect hidden files/dirs:
    - POSIX: names starting with '.'
    - Windows: try FILE_ATTRIBUTE_HIDDEN; fallback to name check
    """
    name = p.name
    if name.startswith(".") and name not in (".", ".."):
        return True

    if os.name == "nt":
        try:
            import ctypes  # type: ignore
            FILE_ATTRIBUTE_HIDDEN = 0x2
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(p))
            if attrs == -1:
                return False
            return bool(attrs & FILE_ATTRIBUTE_HIDDEN)
        except Exception:
            return False

    return False


# --------------------------------------------------------------------------------------
# Cleaning logic
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CleanPlanItem:
    path: Path
    is_dir: bool


def plan_clean_node_dir(node_dir: Path) -> Iterable[CleanPlanItem]:
    """
    Plan deletions inside a node directory: keep only 'settings.xml'.
    Hidden files and directories are removed as well.
    """
    if not node_dir.is_dir():
        return []
    items: list[CleanPlanItem] = []
    for entry in node_dir.iterdir():
        if entry.is_file() and entry.name == ALLOWED_NODE_FILE:
            continue
        items.append(CleanPlanItem(path=entry, is_dir=entry.is_dir()))
    return items


def plan_clean_root_entries(root: Path) -> Iterable[CleanPlanItem]:
    """
    Plan deletions in root:
      - Delete all files (including hidden) except 'workflow.knime'.
      - Delete hidden directories.
      - Keep non-hidden directories (node dirs).
    """
    items: list[CleanPlanItem] = []
    for entry in root.iterdir():
        if entry.is_dir():
            # remove only hidden directories in root
            if is_hidden_path(entry):
                items.append(CleanPlanItem(path=entry, is_dir=True))
            continue

        # Files in root: keep only workflow.knime, remove everything else (hidden included)
        if entry.is_file() and entry.name == ALLOWED_ROOT_FILE:
            continue
        items.append(CleanPlanItem(path=entry, is_dir=False))
    return items


def execute_plan(items: Iterable[CleanPlanItem], *, dry_run: bool, verbose: bool) -> None:
    for it in items:
        if dry_run or verbose:
            print(f"{'DRY-RUN ' if dry_run else ''}REMOVE {'DIR ' if it.is_dir else 'FILE'}: {it.path}", file=sys.stderr)
        if dry_run:
            continue
        try:
            if it.is_dir:
                shutil.rmtree(it.path, ignore_errors=False)
            else:
                # includes regular files, hidden files, symlinks
                it.path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] Failed to remove {it.path}: {e}", file=sys.stderr)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean a KNIME project directory copied under tests/data for knime2py tests."
    )
    p.add_argument("project", nargs="?", help="Workflow NAME under tests/data/<NAME> (omit if using --path).")
    p.add_argument("--path", help="Explicit path to the KNIME project directory.")
    p.add_argument("--data-dir", help="Override tests/data directory (default: <repo>/tests/data).")
    p.add_argument("--dry-run", action="store_true", help="Show what would be deleted, do not delete.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else default_tests_data_dir()
    proj_dir = resolve_project_dir(args.project, args.path, data_dir)

    if not proj_dir.exists():
        print(f"[ERROR] Path does not exist: {proj_dir}", file=sys.stderr)
        return 1
    if not proj_dir.is_dir():
        print(f"[ERROR] Path is not a directory: {proj_dir}", file=sys.stderr)
        return 1
    if is_fs_root(proj_dir):
        print(f"[ERROR] Refusing to operate on filesystem root: {proj_dir}", file=sys.stderr)
        return 2

    # Safety: require workflow.knime to exist at root before cleaning
    wf = proj_dir / ALLOWED_ROOT_FILE
    if not wf.exists():
        print(
            f"[ERROR] '{ALLOWED_ROOT_FILE}' not found in {proj_dir}. "
            f"This does not look like a KNIME project root.",
            file=sys.stderr,
        )
        return 3

    if args.verbose:
        print(f"[INFO] Cleaning project: {proj_dir}", file=sys.stderr)

    plan: list[CleanPlanItem] = []

    # 1) For each immediate subdirectory (node dir), remove everything except settings.xml
    for entry in proj_dir.iterdir():
        if entry.is_dir():
            plan.extend(plan_clean_node_dir(entry))

    # 2) In root, delete all files except workflow.knime, and delete hidden directories
    plan.extend(plan_clean_root_entries(proj_dir))

    # 3) Execute
    execute_plan(plan, dry_run=args.dry_run, verbose=args.verbose or args.dry_run)

    if args.dry_run:
        print("[OK] Dry-run complete.", file=sys.stderr)
    else:
        print("[OK] Cleaning complete.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
