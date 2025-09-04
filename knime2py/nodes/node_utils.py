#!/usr/bin/env python3
from __future__ import annotations

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
