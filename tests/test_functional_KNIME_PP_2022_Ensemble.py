import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_end_to_end_ensemble_scorer():
    """
    Test the end-to-end functionality of the ensemble scorer by generating a Python workbook
    from a KNIME project, executing the workbook, and validating the output score CSV file.

    This function performs the following steps:
    1. Prepares the output directory by cleaning it if it already exists.
    2. Generates the Python workbook from the specified KNIME project.
    3. Executes the generated workbook and checks for successful execution.
    4. Validates the presence and contents of the output score CSV file.
    """
    # Paths
    repo_root = Path(__file__).resolve().parents[1]
    knime_proj = repo_root / "tests" / "data" / "KNIME_PP_2022_Ensemble"
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
    expected_script = out_dir / "KNIME_PP_2022_Ensemble__g01_workbook.py"
    if not expected_script.exists():
        candidates = sorted(out_dir.glob("KNIME_PP_2022_Ensemble*workbook.py"))
        assert candidates, (
            "No generated workbook script found in output dir. "
            f"Contents: {[p.name for p in out_dir.iterdir()]}"
        )
        expected_script = candidates[0]

    # 3) Run the generated workbook.
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
    # 465,183
    # 243,712
    assert len(rows) >= 3, f"Unexpected CSV shape: {rows}"
    assert rows[0] == ["no", "yes"], f"Header mismatch: {rows[0]}"
    assert rows[1] == ["465", "183"], f"First data row mismatch: {rows[1]}"
    assert rows[2] == ["243", "712"], f"Second data row mismatch: {rows[2]}"

