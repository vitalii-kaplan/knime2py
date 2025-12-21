# tests/support/pmml_compare.py
from __future__ import annotations

"""
Utilities for comparing PMML files in tests.

We canonicalize the XML structure to eliminate formatting differences and
assert that both documents share identical tags, attributes, and text.
"""

from pathlib import Path
import difflib
import xml.etree.ElementTree as ET


def _canonical_lines(elem: ET.Element, indent: int = 0) -> list[str]:
    """Produce a canonical list of strings representing the XML tree."""
    pad = "  " * indent
    attrs = " ".join(f'{k}="{(v or "").strip()}"' for k, v in sorted(elem.attrib.items()))
    start = f"{pad}<{elem.tag}"
    if attrs:
        start += f" {attrs}"
    start += ">"
    lines = [start]
    text = (elem.text or "").strip()
    if text:
        lines.append(f"{pad}  {text}")
    for child in list(elem):
        lines.extend(_canonical_lines(child, indent + 1))
    lines.append(f"{pad}</{elem.tag}>")
    return lines


def _canonicalize(path: Path) -> str:
    """Parse and canonicalize the XML at the given path."""
    tree = ET.parse(path)
    root = tree.getroot()
    return "\n".join(_canonical_lines(root))


def compare_pmml(got_path: Path, exp_path: Path) -> None:
    """Assert that two PMML files are equivalent (ignoring formatting)."""
    got_text = _canonicalize(got_path)
    exp_text = _canonicalize(exp_path)
    if got_text != exp_text:
        diff = "\n".join(
            difflib.unified_diff(
                exp_text.splitlines(),
                got_text.splitlines(),
                fromfile=str(exp_path),
                tofile=str(got_path),
                lineterm="",
            )
        )
        raise AssertionError(f"PMML files differ:\n{diff}")
