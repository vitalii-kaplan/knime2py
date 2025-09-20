#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Any

from knime2py.nodes.registry import get_handlers, get_default_handler  # ← updated import


# ----------------------------
# Helpers to read & parse header comments from a module file
# ----------------------------

def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

_ONLY_HASHES_RE = re.compile(r'^\s*#\s*$')
_COMMENT_BORDER_RE = re.compile(r'^\s*#\s*([#=\-\*_~]{6,})\s*$')

def _extract_header_block(src: str) -> List[str]:
    """
    Return a list of raw comment lines from the file header (without leading '#'),
    stopping at the first non-comment line. Blank comment lines are kept as ''.
    Handles an optional shebang and any leading blank lines before the header.
    """
    lines = src.splitlines()
    out: List[str] = []

    i = 0
    # Optional shebang
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1

    # Skip any leading blank lines between shebang and the header comment block
    while i < len(lines) and not lines[i].strip():
        i += 1

    # No header if next non-blank line is not a comment
    if i >= len(lines) or not lines[i].lstrip().startswith("#"):
        return out

    # Collect the contiguous leading comment block (allowing blank lines inside)
    for j in range(i, len(lines)):
        line = lines[j]

        # Stop at first non-comment, non-blank line
        if line.strip() and not line.lstrip().startswith("#"):
            break

        # Preserve blank lines inside the header as paragraph separators
        if not line.strip():
            out.append("")
            continue

        # Comment line
        if _COMMENT_BORDER_RE.match(line) or _ONLY_HASHES_RE.match(line):
            # Treat borders like "########" or similar as blank separators
            out.append("")
            continue

        # Strip leading '#' and a single following space
        body = line.lstrip()[1:]
        if body.startswith(" "):
            body = body[1:]
        out.append(body.rstrip())

    # Normalize: remove leading blanks and collapse consecutive blanks
    while out and out[0] == "":
        out.pop(0)
    cleaned: List[str] = []
    prev_blank = False
    for s in out:
        if s == "":
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(s)
            prev_blank = False
    return cleaned


def _split_paragraphs(comment_lines: List[str]) -> List[List[str]]:
    """
    Split header comment lines into paragraphs by blank lines.
    Each paragraph is a list of non-empty lines (whitespace trimmed).
    """
    paras: List[List[str]] = []
    buf: List[str] = []
    for s in comment_lines:
        if s.strip() == "":
            if buf:
                paras.append(buf)
                buf = []
            continue
        buf.append(s.strip())
    if buf:
        paras.append(buf)
    return paras

def _first_sentence_from_header(paras: List[List[str]]) -> str:
    """
    "Name of the node": use the first non-empty line from the header.
    If that line contains a period, cut at the first period; else return as-is.
    """
    for para in paras:
        for ln in para:
            s = ln.strip()
            if s:
                if "." in s and ":" not in s:
                    return s.split(".", 1)[0].strip()
                return s
    return ""

def _third_paragraph_as_notes(paras: List[List[str]]) -> str:
    """
    Return the 3rd paragraph (index 2) joined with <br>. If not available, return ''.
    """
    if len(paras) >= 3:
        return "<br>".join(html.escape(ln) for ln in paras[2])
    return ""


# ----------------------------
# Fallback utilities (if header absent)
# ----------------------------

def _split_camel(s: str) -> str:
    s = re.sub(r"(\d+)$", "", s)  # drop trailing version digits
    s = re.sub(r"Node\s*Factory$", "", s, flags=re.I)  # drop NodeFactory suffix
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r" \1", s).strip()
    return re.sub(r"\s+", " ", s)

def _humanize_module_name(mod_name: str) -> str:
    base = mod_name.split(".")[-1]
    return base.replace("_", " ").replace("-", " ").strip().title()

def _iter_factory_like_attrs(mod: Any) -> Iterable[str]:
    for attr in dir(mod):
        if not attr:
            continue
        aup = attr.upper()
        if aup.endswith("FACTORY") or aup.endswith("FACTORIES"):
            val = getattr(mod, attr, None)
            if isinstance(val, str) and val.strip():
                yield val.strip()
            elif isinstance(val, (list, tuple)):
                for v in val:
                    if isinstance(v, str) and v.strip():
                        yield v.strip()

def _guess_knime_node_name(mod: Any) -> str:
    for key in ("NODE_NAME", "KNIME_NODE_NAME", "FRIENDLY_NAME", "TITLE", "NAME"):
        v = getattr(mod, key, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    mod_name = getattr(mod, "__name__", "") or ""
    if mod_name:
        nice = _humanize_module_name(mod_name)
        if nice:
            return nice
    for fac in _iter_factory_like_attrs(mod):
        last = fac.rsplit(".", 1)[-1]
        human = _split_camel(last)
        if human:
            return human
    return mod_name or "Unknown Node"


# ----------------------------
# Collect rows (updated for dict registry)
# ----------------------------

def _collect_modules() -> List[Tuple[str, str, str]]:
    """
    Returns rows: (knime_node_name, module_filename, notes_html)
      - knime_node_name: first sentence/line from header comment (fallback to guess)
      - module_filename: basename of module __file__ (e.g., csv_reader.py)
      - notes_html: third paragraph from header (joined with <br>)
    Skips the default fallback/not_implemented handler (FACTORY == "") and
    deduplicates modules that serve multiple FACTORY IDs.
    """
    rows: List[Tuple[str, str, str]] = []

    handlers_map = get_handlers()  # dict: FACTORY -> module
    default_mod = get_default_handler()
    seen_mod_ids = set()

    # Iterate modules (values), skipping the default, and de-dupe
    for fac, mod in handlers_map.items():
        if fac == "":
            continue  # skip the default handler by FACTORY key
        if default_mod is not None and mod is default_mod:
            continue  # skip if same module object was mapped
        mod_id = id(mod)
        if mod_id in seen_mod_ids:
            continue
        seen_mod_ids.add(mod_id)

        mod_name = getattr(mod, "__name__", "")
        if mod_name.endswith(".not_implemented") or getattr(mod, "IS_FALLBACK", False):
            continue  # defensive: skip any explicit fallback module

        # Defaults
        kn_name = _guess_knime_node_name(mod)
        notes_html = ""

        # Header parse if source available
        mod_file = getattr(mod, "__file__", None)
        module_short = "unknown.py"
        if isinstance(mod_file, str) and mod_file:
            module_short = Path(mod_file).name
            src = _read_file(Path(mod_file))
            hdr_lines = _extract_header_block(src)
            paras = _split_paragraphs(hdr_lines)
            name_from_header = _first_sentence_from_header(paras)
            if name_from_header:
                kn_name = name_from_header
            notes_html = _third_paragraph_as_notes(paras) or ""

        rows.append((kn_name, module_short, notes_html))

    rows.sort(key=lambda r: (r[0].lower(), r[1].lower()))
    return rows


# ----------------------------
# HTML render
# ----------------------------

def _render_html(rows: List[Tuple[str, str, str]]) -> str:
    head = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>knime2py — Implemented Nodes</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         padding: 24px; color: #222; background: #fff; }
  h1 { margin: 0 0 12px; font-size: 20px; }
  .meta { color: #666; margin: 0 0 18px; font-size: 13px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #ddd; padding: 8px 10px; vertical-align: top; }
  th { background: #f6f6f6; text-align: left; }
  tr:nth-child(even) td { background: #fafafa; }
  code { background: #f3f3f3; padding: 1px 4px; border-radius: 4px; }
  th.num, td.num { text-align: right; width: 3.25em; color: #666; }
</style>
<body>
<h1>knime2py — Implemented Nodes</h1>
<p class="meta">This page lists the KNIME nodes currently supported for code generation.
For unsupported nodes, the generator produces a best-effort stub and TODOs to guide manual implementation.</p>
<table>
  <thead>
    <tr>
      <th class="num">#</th>
      <th>KNIME Node</th>
      <th>Module</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
"""
    parts = [head]
    for idx, (kn_name, module_short, notes_html) in enumerate(rows, start=1):
        parts.append(
            "    <tr>"
            f"<td class=\"num\">{idx}</td>"
            f"<td>{html.escape(kn_name)}</td>"
            f"<td><code>{html.escape(module_short)}</code></td>"
            f"<td>{notes_html}</td>"
            "</tr>\n"
        )
    parts.append("  </tbody>\n</table>\n</body>\n</html>\n")
    return "".join(parts)



# ----------------------------
# CLI
# ----------------------------

def run_cli(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate an HTML page listing implemented node generators (parsed from header comments).")
    ap.add_argument("--out", type=Path, default=Path("./docs/implemented.html"),
                    help="Output HTML file path (default: ./docs/implemented.html)")
    args = ap.parse_args(argv)

    rows = _collect_modules()
    html_text = _render_html(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
