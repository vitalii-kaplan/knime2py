# docs/_scripts/gen_api.py
from __future__ import annotations
from pathlib import Path
import mkdocs_gen_files

PACKAGE = "knime2py"
SRC_ROOT = Path("src") / PACKAGE

DOC_ROOT = "src"
TOP_LABEL = "Source"

nav = mkdocs_gen_files.Nav()

for path in sorted(SRC_ROOT.rglob("*.py")):
    rel = path.relative_to("src").with_suffix("")      # e.g., knime2py/__main__
    parts = list(rel.parts)

    # Drop package-level __init__.py (keep subpackages as index pages)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            continue

    # Drop __main__.py entirely (we don’t want a docs page for the CLI shim)
    if parts[-1] == "__main__":
        continue

    mod = ".".join(parts)

    # For package pages, write .../index.md so the URL is a directory index
    if (path.name == "__init__.py"):
        doc_path = Path(DOC_ROOT, *parts, "index.md")
    else:
        doc_path = Path(DOC_ROOT, *parts).with_suffix(".md")

    mkdocs_gen_files.open(doc_path, "w").write(
        f"::: {mod}\n"
        "    options:\n"
        "      show_root_heading: true\n"
        "      show_source: true\n"
        "      members_order: source\n"
        "      filters:\n"
        "        - \"!^_\"\n"
    )

    nav[(TOP_LABEL,) + tuple(parts)] = doc_path.as_posix()

with mkdocs_gen_files.open("SUMMARY.md", "w") as f:
    f.writelines(nav.build_literate_nav())
