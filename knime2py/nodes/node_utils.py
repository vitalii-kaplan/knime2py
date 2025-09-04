#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple
from lxml import etree as ET

# ----------------------------
# Generic XML helpers
# ----------------------------

def first(root: ET._Element, xpath: str) -> Optional[str]:
    """Return the first (string) value for xpath, stripped, or None."""
    vals = root.xpath(xpath)
    if vals:
        return (vals[0] or "").strip()
    return None

def all_values(root: ET._Element, xpath: str) -> List[str]:
    """Return all values for xpath as stripped strings."""
    return [(v or "").strip() for v in root.xpath(xpath)]

# ----------------------------
# Regex-based entry iteration
# ----------------------------

_ENTRY_XPATH = ".//*[local-name()='entry']"

def _iter_entries(root: ET._Element):
    """Yield (key, value) pairs for all KNIME <entry key="..." value="..."/> nodes."""
    for ent in root.xpath(_ENTRY_XPATH):
        k = (ent.get("key") or "").strip()
        v = ent.get("value")
        yield k, (v or "").strip() if v is not None else None

def _first_value_re(root: ET._Element, pattern: str, flags=re.I) -> Optional[str]:
    rx = re.compile(pattern, flags)
    for k, v in _iter_entries(root):
        if rx.search(k):
            return v
    return None

def _first_value_re_excluding(root: ET._Element, include_pat: str, exclude_pat: str, flags=re.I) -> Optional[str]:
    inc = re.compile(include_pat, flags)
    exc = re.compile(exclude_pat, flags)
    for k, v in _iter_entries(root):
        if inc.search(k) and not exc.search(k):
            return v
    return None

def _first_value_all_tokens(root: ET._Element, tokens: List[str]) -> Optional[str]:
    """Return first value whose key contains ALL tokens (case-insensitive)."""
    toks = [t.lower() for t in tokens]
    for k, v in _iter_entries(root):
        lk = k.lower()
        if all(t in lk for t in toks):
            return v
    return None

# ----------------------------
# Normalizers
# ----------------------------

def normalize_delim(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    v = raw.strip()
    if len(v) == 1:
        return v
    up = v.upper()
    if up in {"TAB", "\\T", "CTRL-I"}:
        return "\t"
    if up in {"COMMA"}:
        return ","
    if up in {"SEMICOLON", "SEMI", "SC"}:
        return ";"
    if up in {"SPACE"}:
        return " "
    if up in {"PIPE"}:
        return "|"
    if v == "\\t":
        return "\t"
    return v or None

def normalize_char(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    v = raw.strip()
    if v.upper() in {"", "NONE", "NULL"}:
        return None
    if v == "&quot;":
        return '"'
    if v == "&apos;":
        return "'"
    return v[:1] if len(v) >= 1 else None

def looks_like_path(s: str) -> bool:
    if not s:
        return False
    low = s.lower()
    if low.startswith(("file:", "s3:", "hdfs:", "abfss:", "http://", "https://")):
        return True
    if s.endswith(".csv"):
        return True
    if "/" in s or "\\" in s:
        return True
    return False

def bool_from_value(v: Optional[str]) -> Optional[bool]:
    if v is None:
        return None
    t = v.strip().lower()
    if t in {"true", "1", "yes", "y"}:
        return True
    if t in {"false", "0", "no", "n"}:
        return False
    return None

# ----------------------------
# Port helpers / context helpers
# ----------------------------

def normalize_in_ports(in_ports: List[object]) -> List[Tuple[str, str]]:
    """
    Accepts items like ('1393','1') or '1393:1' (or just '1393') and
    returns a normalized list of (src_id, port) as strings.
    """
    norm: List[Tuple[str, str]] = []
    for item in in_ports or []:
        if isinstance(item, tuple) and len(item) == 2:
            src, port = str(item[0]), str(item[1] or "1")
            norm.append((src, port))
        else:
            s = str(item)
            if ":" in s:
                src, port = s.split(":", 1)
                norm.append((src, port or "1"))
            elif s:
                norm.append((s, "1"))
    if not norm:
        norm.append(("UNKNOWN", "1"))
    return norm

def context_assignment_lines(node_id: str, out_ports: List[str]) -> List[str]:
    """
    For reader-like nodes that produce a dataframe named `df`,
    publish it under context keys '<node_id>:<port>'.
    """
    ports = sorted({(p or "1") for p in (out_ports or [])}) or ["1"]
    return [f"context['{node_id}:{p}'] = df" for p in ports]

# ----------------------------
# CSV extractors (regex-driven)
# ----------------------------

def extract_csv_path(root: ET._Element) -> Optional[str]:
    """
    Prefer keys that *sound* like file paths; fall back to any entry value that looks like a path.
    Avoid false-positives like node_file='settings.xml' via looks_like_path().
    """
    # Prefer specific-ish keys first
    for pat in (r"\bpath\b", r"\burl\b", r"\bfile\b", r"location"):
        v = _first_value_re(root, pat)
        if v and looks_like_path(v):
            return v
    # Fallback: any entry value that looks like a CSV/path
    for _k, v in _iter_entries(root):
        if v and looks_like_path(v):
            return v
    return None

def extract_csv_sep(root: ET._Element) -> Optional[str]:
    # cover 'delim', 'separator', 'column_delimiter'
    raw = _first_value_re(root, r"(delim|separator|column[_-]?delimiter)\b")
    return normalize_delim(raw)

def extract_csv_quotechar(root: ET._Element) -> Optional[str]:
    # prefer quote char keys but ignore escape keys
    # exact-ish matches first
    raw = _first_value_re_excluding(root, r"\bquote(_?char)?\b", r"escape")
    if raw is None:
        # looser fallback: any 'quote' key that isn't an escape
        for k, v in _iter_entries(root):
            if "quote" in k.lower() and "escape" not in k.lower():
                raw = v
                break
    return normalize_char(raw)

def extract_csv_escapechar(root: ET._Element) -> Optional[str]:
    raw = _first_value_re(root, r"escape")
    return normalize_char(raw)

def extract_csv_encoding(root: ET._Element) -> Optional[str]:
    # handles 'charset', 'encoding', and writer's 'character_set'
    return (
        _first_value_re(root, r"\bcharacter_set\b")
        or _first_value_re(root, r"\bcharset\b")
        or _first_value_re(root, r"encoding")
    )

def extract_csv_header_reader(root: ET._Element) -> Optional[bool]:
    """
    Reader: look for 'column header', 'hasheader', or plain 'header', but avoid writer-only
    keys like 'write_header'.
    """
    for k, v in _iter_entries(root):
        lk = k.lower()
        if "header" not in lk:
            continue
        if "write" in lk:
            continue
        if "column" in lk or "hasheader" in lk or lk == "header":
            return bool_from_value(v)
    return None

def extract_csv_header_writer(root: ET._Element) -> Optional[bool]:
    """
    Writer: prefer explicit 'writeColumnHeader'/'write_header'; otherwise any key whose
    name contains both 'write' and 'header'.
    """
    v = (
        _first_value_re(root, r"\bwriteColumnHeader\b")
        or _first_value_re(root, r"\bwrite_header\b")
        or _first_value_all_tokens(root, ["write", "header"])
    )
    return bool_from_value(v) if v is not None else None

def extract_csv_na_rep(root: ET._Element) -> Optional[str]:
    """
    Writer NA representation:
      - modern: key='missing_value_pattern' (may be empty string '')
      - older: key contains both 'missing' and 'representation'
    Keep empty string '' as a real value; return None only if not set.
    """
    v = _first_value_re(root, r"^missing_value_pattern$")
    if v is None:
        v = _first_value_all_tokens(root, ["missing", "representation"])
    return v

def extract_csv_include_index(root: ET._Element) -> Optional[bool]:
    raw = _first_value_re(root, r"include[_-]?index")
    return bool_from_value(raw)
