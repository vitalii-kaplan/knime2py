# Tests: Generation, Structure, and Usage

This document summarizes the current state of the **knime2py** test generation, what the generator does to your copied KNIME projects, how the generated tests work, and how to run and tune them.

---

## 1) What the generator does

`test_gen.cli` is a small utility that:

1. **Cleans a copied KNIME project** under `tests/data/<NAME>` (or a `--path` you give it), so only the files needed for reproducible export remain:

   * Inside each **node directory**: keep only `settings.xml`; delete everything else (including hidden files).
   * In the **project root**: keep only `workflow.knime` and non-hidden directories (node dirs); delete all other files and delete hidden directories.
   * Safety checks: refuses to run on filesystem root; requires the presence of `workflow.knime`.

2. **Writes a pytest** into `tests/test_<slug>.py` that:

   * Calls the knime2py CLI to export the workflow to `tests/data/!output` (provided by a fixture).
   * Runs the generated `*_workbook.py`.
   * Compares produced CSV(s) to reference CSV(s) using **relative tolerance** (numeric), with trimming for strings, and strict header and shape checks.

3. **Overwrites by default.** The generated test file is replaced unless you pass `--no-overwrite`.

---

## 2) Directory layout assumptions

```
repo/
├─ src/
│   └─ knime2py/...
├─ tests/
│  ├─ data/
│  │  ├─ <WORKFLOW_NAME>/              # a *copy* of your KNIME project (cleaned by the generator)
│  │  │   └─ workflow.knime
│  │  ├─ data/
│  │  │  └─ <WORKFLOW_NAME>/
│  │  │      ├─ output.csv             # reference tables
│  │  │      ├─ foo_output.csv
│  │  │      └─ bar_output.csv
│  │  └─ !output/                      # produced outputs (managed by fixture)
│  ├─ support/
│  │  └─ csv_compare.py                # shared CSV comparison helper
│  ├─ conftest.py                      # provides output_dir fixture
│  └─ test_<slug>.py                   # generated test(s)
└─ ...
```

* The **reference CSVs** live under `tests/data/data/<WORKFLOW_NAME>/` and can include multiple files; the generator looks for all files matching `*_output.csv`.
* The test writes produced outputs into `tests/data/!output/`.

---

## 3) The comparison helper

All generated tests import a single helper module:

* `tests/support/csv_compare.py`

It provides:

* `compare_csv(got_path, exp_path, rtol=RTOL)` — compares two CSVs.
* `RTOL` — default **relative tolerance** (defaults to `1e-3`, i.e. 0.1%). Can be overridden at runtime via environment variable `K2P_RTOL`.
* `ZERO_TOL` — small **zero tolerance** (defaults to `1e-6`). Any finite numeric value with absolute magnitude `< ZERO_TOL` is treated as `0.0` before comparison in both tables. Can be overridden via `K2P_ZERO_TOL`.

### What “equal” means

* **Headers** must match exactly after trimming.
* **Shape** must match: same row count; each corresponding row has the same number of columns.
* **Numeric cells** are compared using `math.isclose(a, b, rel_tol=RTOL, abs_tol=0)` after mapping finite near-zero values to `0.0` using `ZERO_TOL`.
* `NaN` equals `NaN`. `+∞` and `−∞` must match exactly.
* **Non-numeric cells** must match exactly after trimming.

On failure, the helper prints up to the first 25 mismatches and shows both file paths. Relative errors are reported using the same denominator as `math.isclose` (i.e. `max(|a|, |b|)`), applied after the zero-normalization.

---

## 4) Multiple outputs per workflow

The generated test now supports **multiple output tables**:

* It enumerates all reference files in `tests/data/data/<WORKFLOW_NAME>/**` whose basename matches `*_output.csv`.
* For each reference file `X_output.csv` it expects a produced file with the **same basename** in `tests/data/!output/`.
* Each pair is compared independently via `csv_compare.compare_csv(...)`.

This lets a single workflow test validate several outputs.

---

## 5) The generated test: how it runs

At a high level, each generated test:

1. Uses the `output_dir` fixture (from `conftest.py`) to get a **fresh** `tests/data/!output` directory.
2. Invokes with `PYTHONPATH` pointing at your repo’s `src/` so the CLI resolves local code:

   ```
   python -m knime2py <workflow_dir> --out tests/data/!output --graph off --workbook py
   ```

3. Finds the generated `*_workbook.py` in `!output` and executes it (cwd = `!output`) so relative paths resolve.
4. Compares `!output/<name>_output.csv` files against references under `tests/data/data/<WORKFLOW_NAME>/`.

---

## 6) Tolerances and environment overrides

* Default `RTOL = 1e-3` (0.1%). Override for a run:

  ```
  K2P_RTOL=1e-4 pytest -q
  ```
* Default `ZERO_TOL = 1e-6`. Override:

  ```
  K2P_ZERO_TOL=1e-8 pytest -q
  ```

Use a **larger** `RTOL` when you expect minor, benign numeric drift (e.g., pandas versions, BLAS differences). Use a **smaller** value when you want tighter verification.

`ZERO_TOL` only affects values that are very close to zero; it avoids meaningless relative errors caused by tiny denormalized numbers.

---

## 7) Generator usage

From repo root:

```bash
# Generate test for a workflow copied under tests/data/<NAME>/
python -m test_gen.cli <NAME>

# Or point at an explicit KNIME project directory copy
python -m test_gen.cli --path /abs/path/to/knime_project_copy

# See actions without writing
python -m test_gen.cli <NAME> --dry-run -v

# Keep an existing test file (do not overwrite)
python -m test_gen.cli <NAME> --no-overwrite
```

Defaults:

* `--data-dir` defaults to `<repo>/tests/data`.
* `--tests-dir` defaults to `<repo>/tests`.
* Overwrite **is enabled by default**; add `--no-overwrite` to preserve an existing test file.

Slug rules: test filename is `tests/test_<slug>.py` where `<slug>` keeps alphanumerics and converts others to `_`, collapsing runs.

---


## 8) Running tests

Typical invocations:

```bash
# Run all tests
pytest -q

# Only knime2py roundtrip tests
pytest -q -k roundtrip

# Tighten numeric tolerance for the run
K2P_RTOL=1e-4 pytest -q

# Adjust near-zero normalization
K2P_ZERO_TOL=1e-8 pytest -q

# Show detailed failure output
pytest -vv
```


