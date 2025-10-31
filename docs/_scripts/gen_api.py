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
    rel = path.relative_to("src").with_suffix("")
    parts = list(rel.parts)

    # Drop package-level __init__.py page (keep subpackages as index pages)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            continue

    # Drop __main__.py (don’t generate docs for the CLI shim)
    if parts and parts[-1] == "__main__":
        continue

    mod = ".".join(parts)

    # For package pages, write .../index.md to get nice section URLs
    if path.name == "__init__.py":
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

# Build SUMMARY.md with top-level pages first, then the generated Source tree
with mkdocs_gen_files.open("SUMMARY.md", "w") as f:
    f.write("* [Home](index.md)\n")
    f.write("* [Installation](installation.md)\n")
    f.write("* [Quickstart](quickstart.md)\n")
    f.write("* [UI](ui.md)\n")
    f.writelines(nav.build_literate_nav())
