#!/usr/bin/env python3
"""
knime2py test generator — cleanup + test file creation (relative-tolerant compare)

Usage:
  python -m test_gen.cli <NAME> [--overwrite] [--dry-run] [-v]
  python -m test_gen.cli --path /path/to/knime/project [--overwrite] [--dry-run] [-v]

Options:
  --data-dir PATH     Override the default tests/data directory (default: <repo>/tests/data)
  --tests-dir PATH    Where to write the pytest file (default: <repo>/tests)
  --overwrite         Overwrite an existing generated test file if present
  --dry-run           Show what would be deleted/created, but do not perform actions
  -v, --verbose       Print details
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
      - compares !output/output.csv to data/data/<workflow_name>/output.csv
        using relative numeric tolerance (default RTOL = 2e-4 = 0.02%).
        Override via env var K2P_RTOL.
    """
    slug = slugify(workflow_name).lower()
    template = '''\
# Auto-generated test for KNIME workflow "{workflow_name}"
# Generated by test_gen.cli — do not hand-edit; re-generate from the source workflow.

import csv
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Default relative tolerance: 0.02%. Override by setting K2P_RTOL env var, e.g. K2P_RTOL=1e-4
RTOL = float(os.environ.get("K2P_RTOL", "2e-4"))

def _wipe_dir(p: Path) -> None:
    if p.exists():
        for q in p.iterdir():
            if q.is_dir():
                shutil.rmtree(q, ignore_errors=True)
            else:
                try:
                    q.unlink()
                except FileNotFoundError:
                    pass
    else:
        p.mkdir(parents=True, exist_ok=True)

def _read_csv_rows(path: Path):
    """Read CSV into rows (lists of trimmed strings). Skip fully-empty rows."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        rows = []
        for r in reader:
            if r is None:
                continue
            rr = [(c or "").strip() for c in r]
            if any(rr):  # skip rows that are all empty after trimming
                rows.append(rr)
        return rows

def _try_parse_float(s: str):
    """Return (is_number, value). Accepts inf/-inf/NaN case-insensitively."""
    s2 = (s or "").strip()
    if s2 == "":
        return False, None
    try:
        v = float(s2)
        return True, v
    except Exception:
        low = s2.lower()
        if low in ("nan", "+nan", "-nan"):
            return True, math.nan
        if low in ("inf", "+inf", "infinity", "+infinity"):
            return True, math.inf
        if low in ("-inf", "-infinity"):
            return True, -math.inf
        return False, None

def _cells_equal(a: str, b: str, *, rtol: float) -> bool:
    """Numeric cells equal within RELATIVE tol; strings must match exactly after trimming."""
    an, av = _try_parse_float(a)
    bn, bv = _try_parse_float(b)
    if an and bn:
        # NaN compares equal only if both are NaN
        if math.isnan(av) and math.isnan(bv):
            return True
        # Infinity must match exactly (including sign)
        if math.isinf(av) or math.isinf(bv):
            return av == bv
        # Relative tolerance only (abs_tol = 0)
        return math.isclose(av, bv, rel_tol=rtol, abs_tol=0.0)
    # non-numeric: exact string equality after trim
    return (a or "").strip() == (b or "").strip()

def _compare_csv_with_relative_tolerance(got_path: Path, exp_path: Path, *, rtol: float = RTOL):
    got = _read_csv_rows(got_path)
    exp = _read_csv_rows(exp_path)

    assert len(got) == len(exp), f"Row count differs: got={{len(got)}}, exp={{len(exp)}}"
    assert len(got) > 0, "Empty CSV (no header)"

    # Header must match exactly (after trimming)
    assert got[0] == exp[0], f"Header mismatch:\\nGOT: {{got[0]}}\\nEXP: {{exp[0]}}"

    # All data rows: same number and same width per row
    for i, (gr, er) in enumerate(zip(got, exp)):
        assert len(gr) == len(er), f"Column count differs at row {{i}}: got={{len(gr)}}, exp={{len(er)}}"

    # Compare row-by-row, cell-by-cell with relative tolerance
    mismatches = []
    for i in range(1, len(got)):  # skip header row (0)
        gr, er = got[i], exp[i]
        for j, (ga, eb) in enumerate(zip(gr, er)):
            if not _cells_equal(ga, eb, rtol=rtol):
                if len(mismatches) < 25:
                    an, av = _try_parse_float(ga)
                    bn, bv = _try_parse_float(eb)
                    if an and bn and not (math.isnan(av) and math.isnan(bv)) and not (math.isinf(av) or math.isinf(bv)):
                        # Report relative error exactly like math.isclose uses: denom = max(|a|,|b|)
                        diff = abs(av - bv)
                        denom = max(abs(av), abs(bv))
                        if denom == 0.0:
                            rel = math.inf if diff != 0.0 else 0.0
                        else:
                            rel = diff / denom
                        mismatches.append((i, j, ga, eb, rel))
                    else:
                        mismatches.append((i, j, ga, eb, None))
                else:
                    break
        if len(mismatches) >= 25:
            break

    if mismatches:
        lines = [f"First mismatches (row, col, got, exp, rel_err; uses math.isclose rel_tol={{rtol}}, abs_tol=0):"]
        for m in mismatches:
            i, j, ga, eb, rel = m
            if rel is None or math.isinf(rel):
                lines.append(f"  at ({{i}},{{j}}): got={{ga!r}} exp={{eb!r}}")
            else:
                lines.append(f"  at ({{i}},{{j}}): got={{ga!r}} exp={{eb!r}} rel_err≈{{rel:.8g}}")
        raise AssertionError("\\n".join(lines))

def test_roundtrip_{slug}():
    repo_root = Path(__file__).resolve().parents[1]
    knime_proj = repo_root / "tests" / "data" / "{workflow_name}"
    out_dir = repo_root / "tests" / "data" / "!output"
    expected_csv = repo_root / "tests" / "data" / "data" / "{workflow_name}" / "output.csv"

    # Preconditions
    assert (knime_proj / "workflow.knime").exists(), f"Missing workflow.knime in {{knime_proj}}"
    assert expected_csv.exists(), f"Expected reference CSV missing: {{expected_csv}}"

    # Fresh output dir
    _wipe_dir(out_dir)

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

    _compare_csv_with_relative_tolerance(produced_csv, expected_csv, rtol=RTOL)
'''
    return template.format(workflow_name=workflow_name, slug=slug)

def write_test_file(tests_dir: Path, workflow_name: str, *, overwrite: bool, dry_run: bool, verbose: bool) -> Path:
    tests_dir.mkdir(parents=True, exist_ok=True)
    fname = f"test_{slugify(workflow_name)}.py"
    out = tests_dir / fname
    if out.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing test: {out} (use --overwrite)")
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
    p.add_argument("--overwrite", action="store_true", help="Overwrite an existing generated test file.")
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
            f"[ERROR] '{ALLOWED_ROOT_FILE}' not found in {proj_dir}. "
            f"This does not look like a KNIME project root.",
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

    # Generate pytest file
    try:
        test_path = write_test_file(
            tests_dir, workflow_name, overwrite=args.overwrite, dry_run=args.dry_run, verbose=args.verbose
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
