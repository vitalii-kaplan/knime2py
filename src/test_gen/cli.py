#!/usr/bin/env python3
"""
knime2py test generator — cleanup + test file creation (relative-tolerant compare)

Usage:
  python -m test_gen.cli <NAME> [--no-overwrite] [--dry-run] [-v]
  python -m test_gen.cli --path /path/to/knime/project [--no-overwrite] [--dry-run] [-v]

Options:
  --data-dir PATH     Override the default tests/data directory (default: <repo>/tests/data)
  --tests-dir PATH    Where to write the pytest file (default: <repo>/tests)
  --no-overwrite      Do NOT overwrite an existing generated test file (default: overwrite)
  --dry-run           Show what would be deleted/created, but do not perform actions
  -v, --verbose       Print details

Notes:
- Generated tests import helpers from tests/support/csv_compare.py
- Generated tests rely on the `output_dir` fixture from conftest.py (no local wiping)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

# ------------------------------------------------------------------------------
# Defaults (kept for clarity; generated tests use csv_compare.RTOL or env K2P_RTOL)
# ------------------------------------------------------------------------------
RTOL = 1e-3  # 0.1% (informational; the generated test will read from csv_compare.RTOL)

ALLOWED_NODE_FILE = "settings.xml"
ALLOWED_ROOT_FILE = "workflow.knime"

# --------------------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------------------
def repo_root_from_this_file() -> Path:
    """Assumes this file is at <repo>/src/test_gen/cli.py"""
    return Path(__file__).resolve().parents[2]

def default_tests_data_dir() -> Path:
    return repo_root_from_this_file() / "tests" / "data"

def default_tests_dir() -> Path:
    return repo_root_from_this_file() / "tests"

def is_fs_root(p: Path) -> bool:
    try:
        p = p.resolve()
    except Exception:
        p = p
    return p == p.anchor and p.name == ""

def resolve_project_dir(project: str | None, path: str | None, data_dir: Path) -> Tuple[Path, str]:
    """
    Resolve KNIME project directory and logical NAME.
    If --path is given, use it and take NAME from its basename.
    Else use tests/data/<NAME>.
    """
    if path:
        p = Path(path).expanduser().resolve()
        name = p.name
    else:
        if not project:
            raise SystemExit("ERROR: must provide either <NAME> or --path.")
        p = (data_dir / project).expanduser().resolve()
        name = project
    return p, name

# --------------------------------------------------------------------------------------
# Hidden detection
# --------------------------------------------------------------------------------------
def is_hidden_path(p: Path) -> bool:
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
    """Keep only settings.xml inside a node directory (remove hidden too)."""
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
    In root:
      - delete all files except workflow.knime (hidden included)
      - delete hidden directories
      - keep non-hidden directories (node dirs)
    """
    items: list[CleanPlanItem] = []
    for entry in root.iterdir():
        if entry.is_dir():
            if is_hidden_path(entry):
                items.append(CleanPlanItem(path=entry, is_dir=True))
            continue
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
                it.path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] Failed to remove {it.path}: {e}", file=sys.stderr)

# --------------------------------------------------------------------------------------
# Test file generation
# --------------------------------------------------------------------------------------
def slugify(name: str) -> str:
    """Conservative slug for filenames: keep alnum, convert others to '_' and collapse."""
    s = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return re.sub(r"_+", "_", s) or "workflow"

def render_test_py(workflow_name: str) -> str:
    """
    Produce a pytest that:
      - exports the workflow
      - runs the generated workbook
      - compares produced output.csv to reference output.csv (relative tolerance)
      - uses the `output_dir` fixture from conftest.py for a clean output directory
      - imports the helper module as `from support import csv_compare`
    """
    slug = slugify(workflow_name).lower()
    template = '''\
# Auto-generated test for KNIME workflow "{workflow_name}"
# Generated by test_gen.cli — do not hand-edit; re-generate from the source workflow.

import os
import subprocess
import sys
from pathlib import Path

from support import csv_compare  # provides compare_csv(...) and RTOL

# Resolve RTOL: env K2P_RTOL overrides the library default
_env_rtol = os.environ.get("K2P_RTOL")
RTOL = float(_env_rtol) if _env_rtol is not None else csv_compare.RTOL

def test_roundtrip_{slug}(output_dir: Path):
    repo_root = Path(__file__).resolve().parents[1]
    knime_proj = repo_root / "tests" / "data" / "{workflow_name}"
    out_dir = output_dir  # provided by conftest.py fixture
    expected_csv = repo_root / "tests" / "data" / "data" / "{workflow_name}" / "output.csv"

    # Preconditions
    assert (knime_proj / "workflow.knime").exists(), f"Missing workflow.knime in {{knime_proj}}"
    assert expected_csv.exists(), f"Expected reference CSV missing: {{expected_csv}}"

    # 1) Generate Python workbook(s) only, no graphs
    cmd = [
        sys.executable, "-m", "knime2py",
        str(knime_proj),
        "--out", str(out_dir),
        "--graph", "off",
        "--workbook", "py",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    gen = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root), env=env)
    assert gen.returncode == 0, f"CLI failed\\nSTDOUT:\\n{{gen.stdout}}\\nSTDERR:\\n{{gen.stderr}}"

    # 2) Locate a generated workbook script
    candidates = sorted(out_dir.glob("*_workbook.py"))
    assert candidates, f"No *_workbook.py generated in {{out_dir}}. Contents: {{[p.name for p in out_dir.iterdir()]}}"
    script = candidates[0]

    # 3) Run the generated workbook (cwd=out_dir so relative paths like ../!output/output.csv resolve correctly)
    run = subprocess.run([sys.executable, str(script)], cwd=str(out_dir), capture_output=True, text=True, env=env)
    assert run.returncode == 0, f"Workbook execution failed\\nSTDOUT:\\n{{run.stdout}}\\nSTDERR:\\n{{run.stderr}}"

    # 4) Compare the produced CSV to the expected CSV (with RELATIVE tolerance)
    produced_csv = out_dir / "output.csv"
    assert produced_csv.exists(), f"Produced output.csv not found in {{out_dir}}. Contents: {{[p.name for p in out_dir.iterdir()]}}"

    csv_compare.compare_csv(produced_csv, expected_csv, rtol=RTOL)
'''
    return template.format(workflow_name=workflow_name, slug=slug)

def write_test_file(
    tests_dir: Path,
    workflow_name: str,
    *,
    overwrite: bool = True,
    dry_run: bool,
    verbose: bool,
) -> Path:
    tests_dir.mkdir(parents=True, exist_ok=True)
    fname = f"test_{slugify(workflow_name)}.py"
    out = tests_dir / fname
    if out.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing test: {out} (use --no-overwrite to keep it)")
    if dry_run or verbose:
        print(f"{'DRY-RUN ' if dry_run else ''}WRITE TEST: {out}", file=sys.stderr)
    if not dry_run:
        out.write_text(render_test_py(workflow_name), encoding="utf-8")
    return out

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean a copied KNIME project and generate a pytest that validates knime2py roundtrip (with relative tolerance)."
    )
    p.add_argument("project", nargs="?", help="Workflow NAME under tests/data/<NAME> (omit if using --path).")
    p.add_argument("--path", help="Explicit path to the KNIME project directory.")
    p.add_argument("--data-dir", help="Override tests/data directory (default: <repo>/tests/data).")
    p.add_argument("--tests-dir", help="Where to write the test (default: <repo>/tests).")
    p.add_argument("--no-overwrite", action="store_true", help="Do NOT overwrite existing test file (default: overwrite).")
    p.add_argument("--dry-run", action="store_true", help="Show actions without performing them.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else default_tests_data_dir()
    tests_dir = Path(args.tests_dir).expanduser().resolve() if args.tests_dir else default_tests_dir()
    proj_dir, workflow_name = resolve_project_dir(args.project, args.path, data_dir)

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
            f"[ERROR] '{ALLOWED_ROOT_FILE}' not found in {proj_dir}. This does not look like a KNIME project root.",
            file=sys.stderr,
        )
        return 3

    if args.verbose:
        print(f"[INFO] Cleaning project: {proj_dir}", file=sys.stderr)

    # Plan + execute cleaning
    plan: list[CleanPlanItem] = []
    for entry in proj_dir.iterdir():
        if entry.is_dir():
            plan.extend(plan_clean_node_dir(entry))
    plan.extend(plan_clean_root_entries(proj_dir))
    execute_plan(plan, dry_run=args.dry_run, verbose=args.verbose or args.dry_run)

    # Generate pytest file (overwrite by default)
    try:
        test_path = write_test_file(
            tests_dir,
            workflow_name,
            overwrite=not args.no_overwrite,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 4

    if args.dry_run:
        print(f"[OK] Dry-run complete. Would write: {test_path}", file=sys.stderr)
    else:
        print(f"[OK] Cleaning complete. Test written: {test_path}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
