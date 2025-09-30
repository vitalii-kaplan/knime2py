# tests/support/csv_compare.py
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import List, Tuple

# Default relative tolerance; override at runtime via env var K2P_RTOL
RTOL: float = float(os.environ.get("K2P_RTOL", "1e-3"))  # 0.1%

# Zero tolerance: any finite |value| < ZERO_TOL is treated as 0.0 (in both tables)
ZERO_TOL: float = float(os.environ.get("K2P_ZERO_TOL", "1e-6"))

def read_csv_rows(path: Path) -> List[List[str]]:
    """Read CSV into rows (lists of trimmed strings). Skip fully-empty rows."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        rows: List[List[str]] = []
        for r in reader:
            if r is None:
                continue
            rr = [(c or "").strip() for c in r]
            if any(rr):
                rows.append(rr)
        return rows

def _try_parse_float(s: str) -> Tuple[bool, float | None]:
    """Return (is_number, value). Accepts inf/-inf/NaN case-insensitively."""
    s2 = (s or "").strip()
    if s2 == "":
        return False, None
    try:
        return True, float(s2)
    except Exception:
        low = s2.lower()
        if low in ("nan", "+nan", "-nan"):
            return True, math.nan
        if low in ("inf", "+inf", "infinity", "+infinity"):
            return True, math.inf
        if low in ("-inf", "-infinity"):
            return True, -math.inf
        return False, None

def _normalize_zero(v: float) -> float:
    """Return 0.0 if v is finite and |v| < ZERO_TOL, else v unchanged."""
    if math.isfinite(v) and abs(v) < ZERO_TOL:
        return 0.0
    return v

def cells_equal(a: str, b: str, *, rtol: float) -> bool:
    """
    Numeric cells equal within RELATIVE tol (abs_tol=0). Before comparison,
    any finite |value| < ZERO_TOL is treated as 0.0 in both cells.
    Non-numeric cells must match exactly after trimming.
    """
    an, av = _try_parse_float(a)
    bn, bv = _try_parse_float(b)
    if an and bn:
        # NaN compares equal only if both are NaN
        if math.isnan(av) and math.isnan(bv):
            return True
        # Normalize near-zero values to 0.0 (finite only)
        avz = _normalize_zero(av)
        bvz = _normalize_zero(bv)
        # Infinity must match exactly (including sign)
        if math.isinf(avz) or math.isinf(bvz):
            return avz == bvz
        # Relative tolerance only (abs_tol = 0)
        return math.isclose(avz, bvz, rel_tol=rtol, abs_tol=0.0)
    return (a or "").strip() == (b or "").strip()

def compare_csv(got_path: Path, exp_path: Path, *, rtol: float = RTOL) -> None:
    """Assert two CSVs are equal using relative tolerance for numeric cells, with ZERO_TOL zeroing."""
    got = read_csv_rows(got_path)
    exp = read_csv_rows(exp_path)

    assert len(got) == len(exp), f"Row count differs: got={len(got)}, exp={len(exp)}"
    assert len(got) > 0, "Empty CSV (no header)"
    assert got[0] == exp[0], f"Header mismatch:\nGOT: {got[0]}\nEXP: {exp[0]}"

    for i, (gr, er) in enumerate(zip(got, exp)):
        assert len(gr) == len(er), f"Column count differs at row {i}: got={len(gr)}, exp={len(er)}"

    mismatches = []
    for i in range(1, len(got)):  # skip header
        gr, er = got[i], exp[i]
        for j, (ga, eb) in enumerate(zip(gr, er)):
            if not cells_equal(ga, eb, rtol=rtol):
                if len(mismatches) < 25:
                    an, av = _try_parse_float(ga)
                    bn, bv = _try_parse_float(eb)
                    if an and bn and not (math.isnan(av) and math.isnan(bv)) and not (math.isinf(av) or math.isinf(bv)):
                        # Compute relative error using zero-normalized values
                        avz = _normalize_zero(av)
                        bvz = _normalize_zero(bv)
                        diff = abs(avz - bvz)
                        denom = max(abs(avz), abs(bvz))
                        rel = math.inf if denom == 0.0 and diff != 0.0 else (0.0 if denom == 0.0 else diff / denom)
                        mismatches.append((i, j, ga, eb, rel))
                    else:
                        mismatches.append((i, j, ga, eb, None))
                else:
                    break
        if len(mismatches) >= 25:
            break

    if mismatches:
        lines = [
            f"First mismatches (row, col, got, exp, rel_err; math.isclose rel_tol={rtol}, abs_tol=0; ZERO_TOL={ZERO_TOL}):",
            f"Exp table path: {exp_path}",
            f"Got table path: {got_path}",
        ]
        for i, j, ga, eb, rel in mismatches:
            if rel is None or math.isinf(rel):
                lines.append(f"  at ({i},{j}): got={ga!r} exp={eb!r}")
            else:
                lines.append(f"  at ({i},{j}): got={ga!r} exp={eb!r} rel_err≈{rel:.8g}")
        raise AssertionError("\n".join(lines))
