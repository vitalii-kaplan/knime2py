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
    """Read a CSV file and return its rows as a list of lists of trimmed strings.
    
    This function skips any fully-empty rows in the CSV.
    
    Args:
        path (Path): The path to the CSV file to read.
    
    Returns:
        List[List[str]]: A list of rows, where each row is a list of trimmed strings.
    """
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
    """Attempt to parse a string as a float, handling special cases like NaN and infinity.
    
    Args:
        s (str): The string to parse.
    
    Returns:
        Tuple[bool, float | None]: A tuple where the first element indicates if the parsing was successful,
        and the second element is the parsed float value or None if parsing failed.
    """
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
    """Normalize a value to zero if it is finite and below the defined zero tolerance.
    
    Args:
        v (float): The value to normalize.
    
    Returns:
        float: The normalized value, which will be 0.0 if |v| < ZERO_TOL, otherwise v unchanged.
    """
    if math.isfinite(v) and abs(v) < ZERO_TOL:
        return 0.0
    return v

def cells_equal(a: str, b: str, *, rtol: float) -> bool:
    """Check if two cell values are equal within a specified relative tolerance.
    
    This function treats any finite value less than ZERO_TOL as 0.0 for comparison.
    Non-numeric cells must match exactly after trimming.
    
    Args:
        a (str): The first cell value.
        b (str): The second cell value.
        rtol (float): The relative tolerance for numeric comparison.
    
    Returns:
        bool: True if the cells are considered equal, False otherwise.
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
    """Compare two CSV files for equality, allowing for relative tolerance in numeric cells.
    
    This function asserts that the two CSVs have the same number of rows and that their headers match.
    It checks each cell for equality, treating finite values below ZERO_TOL as 0.0.
    
    Args:
        got_path (Path): The path to the actual output CSV file.
        exp_path (Path): The path to the expected output CSV file.
        rtol (float, optional): The relative tolerance for numeric comparisons. Defaults to RTOL.
    
    Raises:
        AssertionError: If the CSV files differ in row count, header, or any cell values.
    """
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
