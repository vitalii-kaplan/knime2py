#!/usr/bin/env bash
set -euo pipefail

TARGET="src/knime2py/traverse.py"

# Build the multi-line edit text safely.
# Note: read -d '' returns non-zero at EOF; temporarily disable -e.
set +e
IFS= read -r -d '' EDIT_TEXT <<'PROMPT'
Add a **PEP 257–compliant module docstring** to the very top of THIS file and return the FULL updated file. Do not change any code or imports besides inserting/replacing the module docstring.

Requirements:
- Placement: keep any shebang/encoding line first; the module docstring must be the first **statement** immediately after that.
- If a module docstring already exists, replace it; otherwise insert a new one. Keep any existing comment banner (`# ...`) as-is.
- Style: triple double-quotes (`"""`), concise first summary line, then blank line, then structured sections using reStructuredText headings. Wrap lines to ~100 chars. No trailing spaces.

Docstring content (derive all details from this file’s code/constants):
1) Title line: a terse, one-sentence summary of what this module does.
2) Overview: what the module emits/produces and how it fits in the knime2py generator pipeline.
3) Runtime behavior:
   - Inputs (what DataFrame(s) or context keys the generated code reads).
   - Outputs (what is written to `context[...]`, with port mapping and types).
   - Key algorithms or mappings (e.g., sklearn/imbalanced-learn equivalents, column selection rules, stratified logic, collision handling, etc.).
4) Edge cases & safeguards the code implements (e.g., empty/constant columns, NaNs, class imbalance, fallback paths).
5) Generated Code Dependencies: list external libraries actually required by the generated code (pandas, numpy, sklearn, imblearn, matplotlib, lxml, etc.). Mention that these dependensies are dependensies of the generated code, not of this code.
6) Usage:
   - Typical upstream/downstream nodes or how this module is invoked by the emitter.
   - One short example of expected context access (pseudo-code is fine).
7) Node identity (if the file generates code based on `settings.xml`):
   - KNIME factory id(s) (e.g., FACTORY / *_FACTORY constants).
   - Any special flags (e.g., LOOP = "start"/"end") and their meaning.
8) Configuration (if the file generates code based on `settings.xml`):
   - Name the `@dataclass` used for settings and list important fields with brief meaning and defaults.
   - Describe how parse_* functions extract these values (paths/xpaths, fallbacks).

9) Limitations / Not implemented: call out options not supported or approximations vs. KNIME behavior.
10) References:
    - If a `HUB_URL` constant exists, include it.
    - Any relevant KNIME terminology the user should search for.

Formatting example for section headings inside the docstring:

    """Short one-line summary.

    Overview
    ----------------------------
    <text>

    Runtime Behavior
    ----------------------------
    <text>

    Edge Cases
    ----------------------------
    <text>

    Generated Code Dependencies
    ----------------------------
    <text>

    Usage
    ----------------------------
    <text>

    Node Identity
    ----------------------------
    <text>

    Configuration
    ----------------------------
    <text>

    Limitations
    ----------------------------
    <text>

    References
    ----------------------------
    <text>
    """

Acceptance criteria:
- Return ONLY the complete, updated file content (the tool will wrap with markers).
- Do not reorder code, change names, or alter logic.
- Preserve import order and spacing; keep Black/isort-friendly formatting.
PROMPT
set -e

#python -m rag.query_openai_file "$TARGET" --rewrite --edit "$EDIT_TEXT"

#python -m rag.query_openai "Which .py file generates code for KNIME Random Forest node?" 
python -m rag.query_openai "Describe me Runtime Behavior of Decision Tree Learner node knime2py implementation." 
