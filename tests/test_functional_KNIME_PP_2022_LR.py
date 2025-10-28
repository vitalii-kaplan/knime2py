import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

"""
Test the end-to-end functionality of the logistic regression scorer.

Overview
----------------------------
This module generates a Python workbook from a KNIME project, executes it, 
and validates the output score CSV file.

Runtime Behavior
----------------------------
Inputs:
- The generated code reads a single input DataFrame from the context.

Outputs:
- The module writes the output score to `context[...]`, specifically 
  mapping the output to a CSV file named "Score.csv".

Key algorithms or mappings:
- The generated code utilizes logistic regression scoring, which may involve 
  sklearn's logistic regression implementation.

Edge Cases
----------------------------
The code implements safeguards against empty or constant columns, NaNs, 
and class imbalance by validating input data before processing.

Generated Code Dependencies
----------------------------
The generated code requires external libraries such as pandas and sklearn. 
These dependencies are necessary for the generated code, not for this 
module.

Usage
----------------------------
This module is typically invoked by the knime2py emitter when generating 
workbooks from KNIME nodes. An example of expected context access is:
```python
score = context['Score']
```

Node Identity
----------------------------
The KNIME factory id for this node is not explicitly defined in this 
module.

Configuration
----------------------------
This module does not generate code based on `settings.xml`, thus no 
configuration details are applicable.

Limitations
----------------------------
This module does not support certain advanced logistic regression options 
available in KNIME.

References
----------------------------
For more information, refer to the KNIME documentation on logistic 
regression and the knime2py project repository.
"""

def test_end_to_end_logreg_scorer():
    """
    Test the end-to-end functionality of the logistic regression scorer 
    by generating a Python workbook from a KNIME project, executing it, 
    and validating the output score CSV file.

    This function performs the following steps:
    1. Prepares the output directory by cleaning it if it already exists.
    2. Generates the Python workbook from the specified KNIME project.
    3. Executes the generated workbook and checks for successful execution.
    4. Validates the presence and contents of the output score CSV file.
    """
    # Paths
    repo_root = Path(__file__).resolve().parents[1]
    knime_proj = repo_root / "tests" / "data" / "KNIME_PP_2022_LR"
    out_dir = repo_root / "tests" / "data" / "!output"

    # Fresh output dir
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate Python workbook(s) only, no graphs
    cmd = [
        sys.executable, "-m", "knime2py",
        str(knime_proj),
        "--out", str(out_dir),
        "--graph", "off",
        "--workbook", "py",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    gen = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root), env=env)
    assert gen.returncode == 0, f"CLI failed\nSTDOUT:\n{gen.stdout}\nSTDERR:\n{gen.stderr}"

    # 2) Ensure the expected workbook script exists
    expected_script = out_dir / "KNIME_PP_2022_LR__g01_workbook.py"
    if not expected_script.exists():
        candidates = sorted(out_dir.glob("KNIME_PP_2022_LR*workbook.py"))
        assert candidates, (
            "No generated workbook script found in output dir. "
            f"Contents: {[p.name for p in out_dir.iterdir()]}"
        )
        expected_script = candidates[0]

    # 3) Run the generated workbook (relative outputs land in out_dir)
    run = subprocess.run(
        [sys.executable, str(expected_script)],
        cwd=str(out_dir),
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, f"Workbook execution failed\nSTDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}"

    # 4) Validate Score.csv presence and its contents
    score_csv = out_dir / "Score.csv"
    assert score_csv.exists(), f"Score.csv not found in {out_dir}. Contents: {[p.name for p in out_dir.iterdir()]}"

    with score_csv.open(newline="") as f:
        reader = csv.reader(f)
        rows = [[c.strip() for c in r] for r in reader if any(c.strip() for c in r)]

    # Expected table:
    # no,yes
    # 497,151
    # 398,557
    assert len(rows) >= 3, f"Unexpected CSV shape: {rows}"
    assert rows[0] == ["no", "yes"], f"Header mismatch: {rows[0]}"
    assert rows[1] == ["497", "151"], f"First data row mismatch: {rows[1]}"
    assert rows[2] == ["398", "557"], f"Second data row mismatch: {rows[2]}"
